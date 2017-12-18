function my_armijo(f::Function,J::Function,fc,df,xc,pc;maxIter=100, c1=1e-4,b=0.5)
LS = 1
t  = 1
while LS<=maxIter
    if f(xc+t*pc) <= fc + t*c1*dot(df,pc)
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

