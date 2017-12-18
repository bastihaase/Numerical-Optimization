#function to create nxn mesh on the interval given by x and y

function create_mesh(n,x,y)
  res = zeros(n,n,2)
  lx = (x[2]-x[1])/n
  ly = (y[2]-y[1])/n
  for i = 1:n
    for j = 1:n
      res[i,j,:]=[x[1]+(j-0.5)*lx,y[2]-(i-0.5)*ly]
    end
  end
  return res
end

# function to divide interval into n+1 equal pieves

function create_grid(n,x,y)
	res=zeros(n+1)
	h=(y-x)/n
	res[1]=x
	for i=1:n
		res[i+1]=x+i*h
	end
	return res
end
