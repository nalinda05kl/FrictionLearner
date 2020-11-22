
using OrdinaryDiffEq
using ModelingToolkit
using DataDrivenDiffEq
using LinearAlgebra, DiffEqSensitivity, Optim
using DiffEqFlux, Flux
using CSV

using Plots
gr()

PI =  3.1416f0

file_name_θ = "./theta_1m_st0.csv"
file_name_θ̇ = "./omega_1m_st0.csv"
f_θ = CSV.read(file_name_θ; header=false, delim=',', types=fill(Float32,3781))
f_θ̇ = CSV.read(file_name_θ̇; header=false, delim=',', types=fill(Float32,3781))

θ_real_all = convert(Matrix, f_θ)
θ̇_real_all = convert(Matrix, f_θ̇)

# total length and the number of frames
tot_time=62.7
tot_frames=3781
Δt=tot_time/tot_frames

function make_data(θ, θ̇, st, ed)
    data_len = ed-st+1
    start_time = Δt*st
    stop_time = start_time + Δt*(data_len-1)
    tspan = (start_time, stop_time)
    θ_sub = reshape(θ[st:ed], (1,data_len))
    θ̇_sub = reshape(θ̇[st:ed], (1,data_len))
    t_sub = Array(range(start_time, length=data_len, stop=stop_time))
    return θ_sub, θ̇_sub, t_sub, tspan
end

θ_real, θ̇_real, time, tspan = make_data(θ_real_all, θ̇_real_all, 1, 300)
θ_ts, θ̇_ts, time_ts, tspan_ts = make_data(θ_real_all, θ̇_real_all, 300, 3781)

# train data
scatter(time, θ_real', xlabel="time",
    title="Data from the experiment (for training)",
    label="θ_train")
scatter!(time, θ̇_real',
    label="ω_train")

# test data
scatter(time_ts, θ_ts', xlabel="time",
    title="Data from the experiment (for testing)",
    label="θ_test")
scatter!(time_ts, θ̇_ts',
    label="ω_test")

## ODEsolver without inference
g=9.80665f0
l=0.65f0 # not accurate

# ODE function
function pendulum_ode_NLdamp_w_add_par(u, p, t)
    μ₁, μ₂, μ₃ = p
    θ, θ̇ = u
    [θ̇, -μ₃*θ̇^2 - μ₂*θ̇ - μ₁*sin(θ)]
end

# initial conditions and time span
uu0 = [θ_real[1], θ̇_real[1]]

# initial non-optimized parameters
#μ₁=g/l (measured), μ₂=linear-damp, μ₃=non-linear-damp
ppp_ = [g/l, 0.2f0, 0.5f0]

prob= ODEProblem(pendulum_ode_NLdamp_w_add_par, uu0, tspan, ppp_)
sol = solve(prob, Tsit5(), saveat=Δt)

# plot initial solusion
plot(sol,
    title="Comparission of Data and Simulation (par. not opt.)",
    linewidth=2)
scatter!(time, θ_real', xlabel="time",
        label="θ_real")
scatter!(time, θ̇_real',
        label="ω_real")

## Inference
using DifferentialEquations

X_data = vcat(θ_real, θ̇_real)

function loss(p)
    tmp_prob = remake(prob, p=p)
    tmp_sol = solve(tmp_prob, saveat=Δt)
    sum(abs2, Array(tmp_sol) - X_data)
end

pinit = ppp_

function plot_callback(p, l)
    @show l
    tmp_prob = remake(prob, p=p)
    tmp_sol = solve(tmp_prob, saveat=Δt)
    fig = plot(tmp_sol, label="running solution", linewidth=2)
    scatter!(fig, time, X_data', label="data")
    display(fig)
    false
end

println("\nUsing first optimizer:\n")

res1 = DiffEqFlux.sciml_train(
                            loss,
                            pinit,
                            ADAM(0.01),
                            cb=plot_callback,
                            maxiters=100)

println("\nUsing second optimizer:\n")

res2 = DiffEqFlux.sciml_train(loss,
                              res1.minimizer,
                              BFGS(initial_stepnorm=0.01),
                              cb = plot_callback)

println(res2.minimizer)

drag_force = sign.(θ̇_real).*res2.minimizer[3].*(θ̇_real.^2) .+ res2.minimizer[2].*θ̇_real
plot(time, drag_force', xlabel="time", label="friction force (f(aω²+bω))")
scatter!(time, θ̇_real', xlabel="time", label="ω_real")
hline!(time, [0], label="equilibrium")

## Neual ODE inference
# Neural Network  with 2 hidden layers
L = FastChain(
    FastDense(1, 64, tanh),
    FastDense(64, 64, tanh),
    FastDense(64, 64, tanh),
    FastDense(64, 1)
    )

#L = FastChain(
    #FastDense(2, 64, tanh),
    #FastDense(64, 64, tanh),
    #FastDense(64, 64, tanh),
    #FastDense(64, 1)
    #)

p_nn = initial_params(L)
u0 = [θ_real[1], θ̇_real[1]]

α= res2.minimizer[1]

function pendulum_nnode(u, p, t)
    θ, θ̇ = u
    z = L([θ, θ̇], p)
    [θ̇, z[1] - α*sin(θ)]
end

prob_nn = ODEProblem(pendulum_nnode, u0, tspan, p_nn)
sol_nn = solve(prob_nn, Tsit5(), saveat=Δt, dt=Δt)

plot(sol_nn, title="Initial Random Guess")
scatter!(time, θ_real', xlabel="time", label="Θ_real")
scatter!(time, θ̇_real', label="ω_real")

function predict(par)
    Array(solve(prob_nn,
          Vern7(),
          u0=u0, p=par,
          saveat=Δt,
          abstol=1e-6, reltol=1e-6,
          sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP())))
end

X_data = vcat(θ_real, θ̇_real)
X_theta = θ_real

# Prediction (testing!)
pred = predict(p_nn)

function loss(par)
    pred = predict(par)
    sum(abs2, X_data[1,:] .- pred[1,:]), pred
end

# initial loss
loss(p_nn)

# Callback function
loss_list = []
callback(par,l,pred) = begin
    push!(loss_list, l)
    if length(loss_list)%10==0
        println("Current loss after $(length(loss_list)) iterations: $(loss_list[end])")
        tmp_prob = remake(prob_nn, p=par)
        tmp_sol = solve(tmp_prob, saveat=Δt)
        fig = plot(tmp_sol, label="running solution", linewidth=2)
        scatter!(fig, time, X_data', label="data")
        display(fig)
    end
    false
end

println("\nUsing first optimizer:\n")

# First train with ADAM for better convergence
nn_res1 = DiffEqFlux.sciml_train(loss,
                              p_nn,
                              ADAM(0.01),
                              cb=callback,
                              maxiters = 50)

println("\nUsing second optimizer:\n")

# Train with BFGS
nn_res2 = DiffEqFlux.sciml_train(loss,
                            nn_res1.minimizer,
                            BFGS(initial_stepnorm=0.01),
                            cb=callback,
                            maxiters= 50)

# optimized parameters
opt_params_new = nn_res2.minimizer

#using BSON: @save
#@save "drag_model_300_frms_only_n_θ.bson" opt_params

## prediction
using BSON: @load
@load "drag_model_300_frms_only_n_θ.bson" opt_params

opt_params_new = opt_params
# θ_ts, θ̇_ts, time_ts, tspan_ts
#X_train = vcat(θ_real, θ̇_real)
#X_test = vcat(θ_ts, θ̇_ts)

using BSON: @save
@save "dmodel_64_Loss_theta_19_5.bson" opt_params_new

X_data = vcat(θ_real, θ̇_real)
X_test =θ_ts

drag_train = L(θ_real, opt_params_new)
#drag_pred = L(X_test, opt_params)

plot(time, drag_train', xlabel="time(s)", label="F(θ)",
    title="Friction from ANN64(θ), (loss: θ)")
plot!(time, θ_real'/4, label="θ")
hline!(time, [0], label="equilibrium")
savefig("./pivot_figs_4/ann64_Loss_nad_nn_theta_fric_06.png")

#θ̇_plot = Array(range(-3.0, length=600, stop=3.0))
#θ_plot = Array(range(-3.0, length=600, stop=3.0))
#X_plot = hcat(θ_plot, θ̇_plot)
#drag_plot = L(X_plot', opt_params_new)
#plot(θ̇_plot, drag_plot')
#plot(θ_plot, drag_plot')



## some plots for the analysis
drag_force_nn = drag_train

#plot(time, nn_pred_sol', title="Comparission of Data and NN Prediction", linewidth=2)
plot(time, drag_force_nn', xlabel="time", label="drag force from ANN", linewidth=3)
plot!(time, drag_force', xlabel="time", label="drag force w/o ANN", linewidth=3)
scatter!(time, θ_real', xlabel="time", label="θ_real")
scatter!(time, θ̇_real', xlabel="time", label="ω_real")
hline!(time, [0], label="ω at max amplitude")
savefig("./temp_figs/drag_compare_1.png")

signed_drag_train = sign.(θ̇_real).*L(X_data, opt_params)
drag_force_cor = -1*(sign.(θ̇_real).*res2.minimizer[3].*(θ̇_real.^2) .+ res2.minimizer[2].*θ̇_real)

plot(time, signed_drag_train', xlabel="time", label="drag force from ANN", linewidth=3)
plot!(time, drag_force_cor', xlabel="time", label="drag force w/o ANN", linewidth=3)
scatter!(time, θ_real', xlabel="time", label="θ_real")
#scatter!(time, θ̇_real', xlabel="time", label="ω_real")
hline!(time, [0], label="ω at max amplitude")
savefig("./temp_figs/drag_compare_2.png")

plot(time, drag_force_nn', xlabel="time", label="drag force from ANN", linewidth=3)
plot!(time, drag_force', xlabel="time", label="drag force w/o ANN", linewidth=3)
hline!(time, [0], label="ω at max amplitude")
savefig("./temp_figs/drag_compare_2.png")

max_drag = maximum(drag_force_nn)
plot(time, drag_force_nn', xlabel="time", title="Drag force from ANN", label="drag", linewidth=3, legend=:bottomright)
hline!(time, [max_drag], label="max drag")
savefig("./temp_figs/drag_compare_3.png")

# damping vs θ
plot(θ_real', drag_force_nn', linewidth=3, title="Smooth Pendulum", xlabel="θ", ylabel="friction force")
savefig("./temp_figs/drag_vs_theta_1.png")

# damping vs ω
plot(θ̇_real', drag_force_nn', linewidth=3, title="Smooth Pendulum", xlabel="ω (ang. velocity)", ylabel="friction force")
savefig("./temp_figs/drag_vs_omega.png")

drag_θ_0 = zeros(Float32, 300)
drag_θ_half = zeros(Float32, 300)
drag_θ_full = zeros(Float32, 300)

for i = 1:300
    dg = L([0.0, θ̇_real[i]], opt_params)
    dg_half = L([θ_real[1]*0.5, θ̇_real[i]], opt_params)
    dg_full = L([θ_real[1], θ̇_real[i]], opt_params)
    drag_θ_0[i] = dg[1]
    drag_θ_half[i] = dg_half[1]
    drag_θ_full[i] = dg_full[1]
end

plot(θ̇_real', drag_θ_0,
    title="Smooth Pendulum",
    label="When θ = 0.0",
    xlabel="ω (ang. velocity)",
    ylabel="frction force",
    legend=:bottomright)
plot!(θ̇_real', drag_θ_half,
    label="When θ = 0.5 θ₀",
    xlabel="ω (ang. velocity)",
    ylabel="friction force")
plot!(θ̇_real', drag_θ_full,
    label="When θ = θ₀",
    xlabel="ω (ang. velocity)",
    ylabel="friction force")
savefig("Figures_pendulum_3/drag_vs_omega_w_theta_fixed.png")
