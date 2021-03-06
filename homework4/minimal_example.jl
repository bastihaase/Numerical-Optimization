using GridInterpolations
using PyPlot

include("check_derivative.jl")

l=[(0.0001*x).^2 for x in 1:100000]
dl=[2*0.0001*x for x in 1:100000]
grid=[0.000h1*x for x in 1:100000]

h(t)=interpolate(RectangleGrid(grid),l,[t])
dh(t)=interpolate(RectangleGrid(grid),dl,[t])

check_derivative_at_point(h,dh,1,1,[3.0])

#plot([0.002*x for x in 1:5000],[h(0.002*x) for x in 1:5000])

