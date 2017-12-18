using PyPlot

function create_plot_data(v,f,q,x)
  n=size(v)[1]
  res=zeros(n)
  for i= 1:n
    res[i]=abs( (f(v[i]+1)-x)/(f(v[i])-x)^q )
  end
  return res
end

#Examples
f(k)=1+(0.5)^(2.0^k)
g(k)=1/factorial(k)
h(k)=(0.25)^(2^(k-1))/k*mod(k,2) +(0.25)^(2^k)*(1-mod(k,2))
r(k)=1/k

#Plots
inp=map(x->x,[1,2,3,4,5,6,7,8,9,10])
out=create_plot_data(inp , f, 1, 1)
loglog(inp,out)
out=create_plot_data(inp , f, 2, 1)
loglog(inp,out)
out=create_plot_data(inp,g,1,0)
loglog(inp,out)
out=create_plot_data(inp,h,2,0)
loglog(inp,out)
out=create_plot_data(inp,h,1,0)
loglog(inp,out)
out=create_plot_data(inp,r,1,0)
loglog(inp,out)


#Estimating q

#checks if the quotient increases

function is_increasing(v)
  n=size(v)[1]
  for i = 1:(n-1)
    if v[i]>v[i+1]
      return false
    end
  end
  return true
end

#checks if the sequence is q convergent, paramters may need tweaking for sequences in the future

function test_if_q_conv(f,q,x,iter=10,lambda=x->x)
  list=map(lambda,[1:iter])
  quot=zeros(iter)
  for i= 1:iter
    quot[i]=abs( (f(list[i]+1)-x))/abs((f(list[i])-x))^q
  end
  if (maximum(quot)-minimum(quot)>10) || is_increasing(quot)
    return false
  end
  return true
end

#tries to find the maximal q for which a sequence is q convergent

function find_q(f,q_start,x,iter=10,lambda=x->x)
  q=q_start
  while test_if_q_conv(f,q,x,iter,lambda) && q<20
    q=q^2
  end
  i=1
  while  (test_if_q_conv(f,q,x,iter,lambda)==false ) && i<20
    q=sqrt(q)
    i+=1
  end
  return q
end

#checks if function converges superlinearly

function converges_superlinearly(f,x,iter=10,lambda=x->x)
  list=map(lambda,[1:iter])
  quot=zeros(iter)
  for i= 1:iter
    quot[i]=abs( (f(list[i]+1)-x))/abs((f(list[i])-x))
  end
  if quot[iter]<quot[1]
    return true
  end
  return false
end

#approximates Q-convergence
find_q(f,1.1,1)
find_q(g,1.1,0)
find_q(h,1.1,0)
find_q(r,1.1,0)

#"determines" whether seq converges superlinearly
converges_superlinearly(f,1,5)
converges_superlinearly(g,0)
converges_superlinearly(h,0,5)
converges_superlinearly(r,0)
