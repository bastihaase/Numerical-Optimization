function convert_matrix_vector(A)
  if size(A)[1]==1
    n=size(A)[2]
    ret=zeros(n)
    for i = 1:n
      ret[i]=A[1,i]
    end
    return ret
  end
  n=size(A)[1]
  ret=zeros(n)
  for i= 1:n
      ret[i]=A[i,1]
  end
  return ret
end


function gd(f::Function,J::Function,x::Vector;maxIter=10000,atol=1e-6,doPrint=true)

    his = zeros(maxIter,2)
    n = length(x)
    I = eye(n)

    i = 1
    xOld = x
    df   = J(x)

    while i<=maxIter
        fc = f(x)[1,1]
        his[i,:] = [norm(fc) norm(df)]
        # get search direction
        pk    = - df
        # line search
        ak,LS = armijo(f,fc,df,x,pk,maxIter=10000)
        if doPrint
            @printf "iter=%04d\t|f|=%1.2e\t|df|=%1.2e\tLS=%d\n" i his[i,1] his[i,2] LS
        end
		    if LS==-1
			      @printf "Linesearch failed!"
            his = his[1:i,:]
			      break;
		    end

        # update x and H
        x    += ak*pk
        if i<5 || mod(i,9)==0
          plot(x[1:9],x[10:18],marker="o")
        end
        df  = J(x)
        if(norm(df)<atol)
            his = his[1:i,:]
            break
        end
        i+=1
    end
    return x,his
end



function armijo(f::Function,fc,df,xc,pc;maxIter=30, c1=1e-4,b=0.5)
  LS = 1
  t  = 1
  while LS<=maxIter
    if f(xc+t*pc)[1,1] <= (fc + t*c1*dot(convert_matrix_vector(df),convert_matrix_vector(pc)))
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


