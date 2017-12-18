function my_armijo(f::Function,J::Function,fc,df,xc,pc;maxIter=10000, c1=1e-4,b=0.5)
LS = 1
t  = 1
while LS<=maxIter
    if f(xc+t*pc)[1] <= fc[1] + t*c1*dot(df,pc)[1]
        break
    end
    t *= b
    LS += 1
end
if LS>maxIter
	LS= -1
	t = 0.0
end
return t,LS
end

