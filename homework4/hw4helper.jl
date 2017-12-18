# HINT: reading a matlab file can be done like this:
#   using MAT
#   vars = matread("hw4data.mat")
#   data = vars["data"]


# linear interpolation of 1D data given on a nodal discretization of tspan into M intervals
function linearInter(u,tspan,M,tp)
    h   = vec(tspan[2:2:end]-tspan[1:2:end])./(M-1)
    N   = Int(length(tp))
    ut  = zeros(N)
    ti = zero(Float64)
    ul = zero(Float64)
    ur = zero(Float64)
    idx = zero(Int)
    for i=1:N
        ti  = (tp[i]-tspan[1])/h[i] +1
        idx = Int(floor(ti))
        if (idx >= 0) && (idx <= M)
                ti -= idx
                ul = (idx  >=1) ? u[idx]   : 0.0
                ur = (idx  <M)  ? u[idx+1] : 0.0
                ut[i] = ul + (ur-ul)*ti  # (1 - yi)*Tl + yi * Tr
        end
    end
    return ut
end
