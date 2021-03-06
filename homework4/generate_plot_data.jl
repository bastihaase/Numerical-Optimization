include("mesh.jl")
xgrid=create_grid(25,0.1,5)
ygrid=create_grid(25,0.1,5)
zgrid=zeros(26,26)
for i = 1:26
	for j = 1:26
		zgrid[i,j]=log(f([xgrid[i],ygrid[j]])[1])
	end
end


include("mesh.jl")
xgrid=create_grid(100,4.0,11.5)
ygrid=create_grid(100,-1.0,6.5)
zgrid=zeros(101,101)
for i = 1:101
	for j = 1:101
		zgrid[i,j]=sqrt(Freudenstein([xgrid[i],ygrid[j]])[1])
	end
end


include("mesh.jl")
xgrid=create_grid(100,-6.0,6.0)
ygrid=create_grid(100,-6.0,6.0)
zgrid=zeros(101,101)
for i = 1:101
	for j = 1:101
		zgrid[i,j]=xgrid[i]*ygrid[j]
	end
end
