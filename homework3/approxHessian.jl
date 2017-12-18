function approx_action_of_hessian(grad,x,p,eps=10.0^(-6))
  return (grad(x+eps*p)-grad(x))/eps
end

function approx_hessian(grad,x,eps=10.0^(-6))
  n=size(x)[1]
  hess=zeros(n,n)
  for i = 1:n
    p=zeros(n)
    p[i]=1
    hess[:,i]=approx_action_of_hessian(grad,x,p,eps)
  end
  return hess
end
