using OrdinaryDiffEq
using ModelingToolkit
using DataDrivenDiffEq
using LinearAlgebra, DiffEqSensitivity, Optim
using DiffEqFlux, Flux
using CSV
using DifferentialEquations
using Plots
gr()

PI =  3.1416f0

## Data Simulation

tspan = (0.0, 10.0) # time span
u₀ = [0.65, 0.0] # initial conditions θ₀, θ̇₀
Δt = 0.01 # time step

θᴱ=0.1 #0.015 # off-set for the angle
g=9.8 # accelaration due to gravity
L=0.64 # length of the pendulum
μ=0.6 # coefficiant of static friction
R=0.01 # radius of the circular hinge
b=1.8e-5 # b=6*π*rb*η
m=0.2 # mass (in kg)

α₀ = g/L
α₁ = b/m
α₂ = μ*R/L
p = [α₀, α₁, α₂]

## friction functions
function air_drag(θ̇, α₁)
    return α₁*θ̇
end

function pivot_friction(θ, θ̇, α₀, α₂)
    return α₂*(α₀*cos(θ) + θ̇^2)*sign(θ̇)
end

function pivot_friction_with_os(θ, θ̇, θᴱ, α₀, α₂)
    return α₂*(α₀*cos(θ+θᴱ) + θ̇^2)*sign(θ̇)
end

function pivot_friction_with_os_asym(θ, θ̇, θᴱ, α₀, α₂, Ω⁺, Ω⁻)
    return (α₂*(α₀*cos(θ+θᴱ) + θ̇^2)*sign(θ̇))*(Ω⁺ + Ω⁻*sign(θ))
end

## ODE functions
function real_pendulum(u, p, t)
    θ, θ̇ = u
    α₀, α₁, α₂ = p
    [θ̇, - α₀*sin(θ) - pivot_friction(θ, θ̇, α₀, α₂) - air_drag(θ̇, α₁)]
end

function real_pendulum_os(du, u, p, t)
    θ, θ̇ = u
    α₀, α₁, α₂ = p
    du[1] = θ̇
    du[2] = - (α₀*(sin(θ)*cos(θᴱ) + cos(θ)*sin(θᴱ))
              + pivot_friction_with_os(θ, θ̇, θᴱ, α₀, α₂)
              + air_drag(θ̇, α₁))
    nothing
end

function real_pendulum_os_asym(du, u, p, t)
    θ, θ̇ = u
    α₀, α₁, α₂ = p
    Ω⁺, Ω⁻ = 1.0, 1.0
    du[1] = θ̇
    du[2] = - (α₀*(sin(θ)*cos(θᴱ) + cos(θ)*sin(θᴱ))
              + pivot_friction_with_os_asym(θ, θ̇, θᴱ, α₀, α₂, Ω⁺, Ω⁻)
              + air_drag(θ̇, α₁))
    nothing
end

function real_pendulum_os_var_mu(du, u, p, t)
    θ, θ̇ = u
    α₀, α₁, α₂ = p
    vα₂ = abs(θ)*α₂
    du[1] = θ̇
    du[2] = - (α₀*(sin(θ)*cos(θᴱ) + cos(θ)*sin(θᴱ))
              + pivot_friction_with_os(θ, θ̇, θᴱ, α₀, vα₂)
              + air_drag(θ̇, α₁))
    nothing
end

prob= ODEProblem(real_pendulum_os, u₀, tspan, p)
sol= solve(prob , saveat=Δt)
θ_os = sol[1,:]
θ̇_os = sol[2,:]
plot(sol.t, θ_os, label="θ_os", xlabel="time (s)",
       title="SIM. DATA")
plot!(sol.t, θ̇_os, label="ω_os")

prob= ODEProblem(real_pendulum_os_asym, u₀, tspan, p)
sol= solve(prob , saveat=Δt)
θ_asym = sol[1,:]
θ̇_asym = sol[2,:]
plot(sol.t, θ_asym, label="θ_os_asym", xlabel="time (s)",
       title="SIM. DATA")
plot!(sol.t, θ̇_asym, label="ω_os_asym")

prob= ODEProblem(real_pendulum_os_var_mu, u₀, tspan, p)
sol= solve(prob , saveat=Δt)
θ_vm = sol[1,:]
θ̇_vm = sol[2,:]
plot(sol.t, θ_vm, label="θ_os_var_mu", xlabel="time (s)",
       title="SIM. DATA")
plot!(sol.t, θ̇_vm, label="ω_os_var_mu")

## Friction analysis

# symetric fiction
θ, θ̇ = θ_os, θ̇_os
f_os = (
       α₀.*cos.(θ)*sin.(θᴱ)
       .+ α₂.*(α₀.*cos.(θ) + θ̇.^2).*sign.(θ̇)
       .+ α₁.*θ̇
       )

plot(f_os,
     title="sym. friction with offset",
     xlabel="time steps",
     label="Fric. from ODE",
     linewidth=2)
plot!(θ, label="θ")
hline!([0], label="equilibrium")
savefig("./pivot_figs_4/sim_fric_sym_1.png")

# step friction
θ, θ̇ = θ_asym, θ̇_asym
Ω⁺, Ω⁻ = 1.0, 1.0
f_asym = (
         α₀.*cos.(θ)*sin.(θᴱ)
         .+ (α₂.*(α₀.*cos.(θ.+θᴱ) .+ θ̇.^2).*sign.(θ̇)).*(Ω⁺ .+ Ω⁻.*sign.(θ))
         .+ α₁.*θ̇
         )

plot(f_asym,
     title="step. friction with offset",
     xlabel="time steps",
     label="Fric. from ODE",
     linewidth=2)
plot!(θ, label="θ")
hline!([0], label="equilibrium")
savefig("./pivot_figs_4/sim_fric_step_1.png")

# variable μ
θ, θ̇ = θ_vm, θ̇_vm
vα₂ = abs.(θ).*α₂
f_vm = (
       α₀.*cos.(θ)*sin.(θᴱ)
       .+ vα₂.*(α₀.*cos.(θ) + θ̇.^2).*sign.(θ̇)
       .+ α₁.*θ̇
       )
plot(f_vm,
     title="θ dep. μ: friction with offset",
     xlabel="time steps",
     label="Fric. from ODE",
     linewidth=2)
plot!(θ, label="θ")
hline!([0], label="equilibrium")
savefig("./pivot_figs_4/sim_fric_vm_1.png")

# variable μ
vα₂ = 0.05*abs.(θ).*α₂
f_vm = (
       α₀.*cos.(θ)*sin.(θᴱ)
       .+ vα₂.*(α₀.*cos.(θ) + θ̇.^2).*sign.(θ̇)
       .+ α₁.*θ̇
       )
plot(f_vm,
     title="θ dep. μ: friction with offset",
     xlabel="time steps",
     label="Fric. from ODE",
     linewidth=2)
plot!(θ, label="θ")
hline!([0], label="equilibrium")
savefig("./pivot_figs_4/sim_fric_kvm_1.png")


## Under construction ↓

α₀ = g/L
α₁ = b/m
α₂ = μ*R/L
θ=θ_sol
θ̇=θ̇_sol

fric = (
       .+ α₂.*(α₀.*cos.(θ) + θ̇.^2).*sign.(θ̇)
       .+ α₁.*θ̇
       )

fric_os =  (
           α₀.*cos.(θ).*sin(θᴼ)
           .+ α₂.*(α₀.*cos.(θ.+θᴼ) .+ θ̇.^2).*sign.(θ̇)
           .+ α₁.*θ̇
           )

fric_os_asym =  (
                α₀.*cos.(θ).*sin(θᴼ)
                .+ abs.(θ).*α₂.*(α₀.*cos.(θ.+θᴼ) .+ θ̇.^2).*sign.(θ̇)
                .+ α₁.*θ̇
                )

plot(fric)

plot!(fric_os,
     title="Asym. friction with offset",
     xlabel="time steps",
     label="Fric. from ODE",
     linewidth=2)

plot!(fric_os_asym,
     title="Asym. friction with offset",
     xlabel="time steps",
     label="Fric. from ODE",
     linewidth=2)

plot!(θ_sol, label="θ_os")

plot(θ_sol, θ̇_sol, fric_os_asym, st = [:surface])

plot(θ_sol, fric_os_asym)

scatter(θ̇_sol, fric_os_asym)

plot(θ_sol, θ̇_sol)

function fric_from_ode_func(θ̇; θ=u₀[1])
    (
    α₀.*cos.(θ).*sin(θᴼ)
    .+ 5*abs.(θ).*α₂.*(α₀.*cos.(θ.+θᴼ) .+ θ̇.^2).*sign.(θ̇)
    .- α₁.*θ̇
    )
end

#function fric_from_ode_func(θ̇; θ=u₀[1])
    #(
    #- (α₀.*cos.(θ).*sin(θᴼ)
    #.+ α₂.*(α₀.*cos.(θ.+θᴼ) .+ θ̇.^2).*sign.(θ̇)).*(0.5 .+ 0.25.*sign.(θ))
    #.- α₁.*θ̇
    #)
#end

f₁(θ̇) = fric_from_ode_func(θ̇, θ=0.0)
f₂(θ̇) = fric_from_ode_func(θ̇, θ=u₀[1]/2)
f₃(θ̇) = fric_from_ode_func(θ̇, θ=u₀[1])

plot(f₁, -5, 5, xlabel="ω", ylabel="friction", label="F(ω, θ=0)", title="Friction vs ω")
plot!(f₂, -5, 5, label="F(ω, θ=θ₀/2)")
plot!(f₃, -5, 5, label="F(ω, θ=θ₀)")
#savefig("./pivot_figs_2/Fric_ODE_vs_omega_s.png")

function fric_from_ode_func_wo_sign(θ̇; θ=u₀[1])
    (
    - (μ*R*g/(L*L)).*cos.(θ)
    - (μ*R/L).*θ̇.*θ̇
    - (b/m).*θ̇
    )
end

ff₁(θ̇) = fric_from_ode_func_wo_sign(θ̇, θ=0.0)
ff₂(θ̇) = fric_from_ode_func_wo_sign(θ̇, θ=u₀[1]/2)
ff₃(θ̇) = fric_from_ode_func_wo_sign(θ̇, θ=u₀[1])

plot(ff₁, -5, 5, xlabel="ω", ylabel="friction", label="F(ω, θ=0)", title="Friction vs ω for fixed θ [with out sign( )]")
plot!(ff₂, -5, 5, label="F(ω, θ=θ₀/2)")
plot!(ff₃, -5, 5, label="F(ω, θ=θ₀)")
savefig("./pivot_figs_2/Fric_ODE_vs_omega_ns.png")

## Neural ODE to find the friction function

# Data to solve the ODE inference problem
X_data = vcat(θ_sol', θ̇_sol')

# Data to train the neural network
X_data_2 = vcat(θ_sol', sign.(θ̇_sol)')

# Data to train the neural network
X_data_3 = vcat(θ_sol', θ̇_sol', sign.(θ̇_sol)')
# Input (1) : only θ
#L1 = FastChain(
    #FastDense(1, 64, tanh),
    #FastDense(64, 64, tanh),
    #FastDense(64, 64, tanh),
    #FastDense(64, 1)
    #)

# Input (2) : θ and signof(θ̇)
L1 = FastChain(
    FastDense(2, 6, tanh),
    FastDense(6, 6, tanh),
    FastDense(6, 6, tanh),
    FastDense(6, 6, tanh),
    FastDense(6, 1)
    )

#L1 = FastChain(
    #FastDense(2, 64, tanh),
    #FastDense(64, 64, tanh),
    #FastDense(64, 1)
    #)

#L2 = FastChain(
    #FastDense(2, 64, tanh),
    #FastDense(64, 64, tanh),
    #FastDense(64, 1)
    #)

p_nn = initial_params(L1)

#function nnode_pendulum(u, p, t)
    #θ, θ̇ = u
    #sn=sign(θ̇)
    #z = L1([θ,θ̇,sn], p)
    #[θ̇, -(g/L)*sin(θ) - z[1]]
#end

function nnode_pendulum(u, p, t)
    θ, θ̇ = u
    z = L1([θ, θ̇], p)
    [θ̇, -(g/L)*sin(θ) - z[1]]
end

prob_nn = ODEProblem(nnode_pendulum, u₀, tspan, p_nn)
sol_nn = solve(prob_nn, Tsit5(), saveat=Δt)

plot(sol_nn, label="prediction", xlabel="time (s)", title="Initial Random Guess from NNODE [2 In L1]")
scatter!(sol.t, θ_sol, label="Θ_data")
scatter!(sol.t, θ̇_sol, label="ω_data")
savefig("./pivot_figs_2/infer_NNODE_train_init_guess_s_2InL1.png")

function predict(par)
    Array(solve(prob_nn,
          Vern7(),
          u0=u₀, p=par,
          saveat=Δt,
          abstol=1e-6, reltol=1e-6,
          sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP())))
end

#function loss(par)
    #pred = predict(par)
    #sum(abs2, X_data .- pred), pred
#end

function loss(par)
    pred = predict(par)
    sum(abs2, X_data[1,:] .- pred[1,:]), pred
end


# initial loss
loss(p_nn)

loss_list = []
callback(par,l,pred) = begin
    push!(loss_list, l)
    println("Current loss after $(length(loss_list)) iterations: $(loss_list[end])")
    #if length(loss_list)%49==0
        #tmp_prob = remake(prob_nn, p=par)
        #tmp_sol = solve(tmp_prob, saveat=Δt)
        #fig = plot(tmp_sol, label="running solution", linewidth=2)
        #scatter!(fig, sol.t, X_data', label="data")
        #display(fig)
    #end
    false
end

println("\nUsing first optimizer:\n")

# First train with ADAM for better convergence
nn_res1 = DiffEqFlux.sciml_train(loss,
                              p_nn,
                              ADAM(0.01),
                              cb=callback,
                              maxiters = 100)

println("\nUsing second optimizer:\n")

# Train with BFGS
nn_res2 = DiffEqFlux.sciml_train(loss,
                            nn_res1.minimizer,
                            BFGS(initial_stepnorm=0.01),
                            cb=callback)

nn_opt_pars = nn_res1.minimizer
#nn_opt_pars = nn_res2.minimizer

using BSON: @save
@save "pivot_model_theta_sign_2InL1.bson" nn_opt_pars

nn_pred_sol = predict(nn_opt_pars)

# ploting loss
#plot(loss_list, yaxis = :log, xaxis = :log, xlabel = "Iterations", ylabel = "Loss")
plot(loss_list, yaxis = :log, xaxis = :log, xlabel = "Iterations", ylabel = "Loss", title="Loss [ NN with in_1: θ, In_2: ω L1]")
savefig("./pivot_figs_2/loss_model_s_2InL1.png")

# Plot the predictin solution
plot(sol.t, nn_pred_sol', label="pred", title="Sim. Data and NNODE Prediction [2InL1]", linewidth=2)
scatter!(sol.t, θ_sol, xlabel="time (s)",
        label="θ_data")
scatter!(sol.t, θ̇_sol,
        label="ω_data")
savefig("./pivot_figs_2/init_sol_nnode_s_2InL1.png")

fric_from_nn = L1(X_data, nn_opt_pars)

plot(sol.t, -1*(fric_from_nn'), ylabel="friction", xlabel="time (s)", label="friction from ANN", title="NN trained with θ and ω, Loss is based on θ", linewidth=3)
plot!(sol.t, fric_from_ode, xlabel="time", label="friction from ODE", linewidth=3)
plot!(sol.t, θ_sol/2.0, label="θ/2")
plot!(sol.t, θ̇_sol/4.0, label="ω/4")
savefig("./pivot_figs_2/friction_vs_time_s_2InL1.png")

#plot(sol.t, -1*(fric_from_nn'), ylabel="friction", xlabel="time (s)", label="friction from ANN", title="trained with only θ", linewidth=3)
#plot!(sol.t, fric_from_ode, xlabel="time", label="friction from ODE", linewidth=3)
#savefig("./pivot_figs_2/friction_vs_time_ns_2.png")

plot(θ_sol, -1*(fric_from_nn'), xlabel="θ", ylabel="friction",
     label="NNODE with 2In",
     title="Friction vs θ", linewidth=3)
plot!(θ_sol, fric_from_ode, linewidth=3, label="ODE with 3In")
savefig("./pivot_figs_2/friction_vs_theta_s_2In.png")

#plot(θ_sol, -1*(fric_from_nn'), xlabel="θ", ylabel="friction",
     #label="NNODE with out sign( )",
     #title="Friction vs θ", linewidth=3)
#savefig("./pivot_figs_2/friction_vs_theta_ns2.png")

plot(θ̇_sol, -1*(fric_from_nn'), xlabel="ω", ylabel="friction",
     label="NNODE with 2In",
     title="Friction vs ω", linewidth=3)
plot!(θ̇_sol, fric_from_ode, linewidth=3, label="ODE with 3In")
savefig("./pivot_figs_2/friction_vs_omega_s_2In.png")

#plot(θ̇_sol, -1*(fric_from_nn'), xlabel="ω", ylabel="friction",
     #label="NNODE with out sign( )",
     #title="Friction vs ω", linewidth=3)
#savefig("./pivot_figs_2/friction_vs_omega_ns2.png")

#L1([Array([0.2, 0.3]), Array([-1,1])'], nn_opt_pars)

function fric_NN(θ, θ̇)
    out = -L1([θ, θ̇], nn_opt_pars)
    out[1]
end

function fric_from_nnode_func(θ̇; θ=u₀[1])
    fric_NN(θ, θ̇)
end

fN₁(θ̇) = fric_from_nnode_func(θ̇, θ=0.0)
fN₂(θ̇) = fric_from_nnode_func(θ̇, θ=u₀[1]/2)
fN₃(θ̇) = fric_from_nnode_func(θ̇, θ=u₀[1])

plot(fN₁, -5, 5, xlabel="ω", ylabel="friction", label="F(ω, θ=0)", title="Friction (from NNODE) vs ω for fixed θ [with 2In]")
plot!(fN₂, -5, 5, label="F(ω, θ=θ₀/2)")
plot!(fN₃, -5, 5, label="F(ω, θ=θ₀)")
savefig("./pivot_figs_2/Fric_NNODE_vs_omega_s_2In.png")

function fric_from_nnode_func2(θ; θ̇=u₀[2])
    fric_NN(θ, θ̇)
end

ffN₁(θ) = fric_from_nnode_func2(θ, θ̇=0.0)
ffN₂(θ) = fric_from_nnode_func2(θ, θ̇=u₀[2]/2)
ffN₃(θ) = fric_from_nnode_func2(θ, θ̇=u₀[2])

plot(ffN₁, -5, 5, xlabel="θ", ylabel="friction", label="F(θ, ω=0)", title="Friction (from NNODE) vs θ for fixed ω [2In]")
plot!(ffN₂, -5, 5, label="F(θ, ω=ω₀/2)")
plot!(ffN₃, -5, 5, label="F(θ, ω=ω₀)")
savefig("./pivot_figs_2/Fric_NNODE_vs_theta_s_2In.png")
