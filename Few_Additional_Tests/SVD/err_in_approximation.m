function [err]=err_in_approximation(AB,AB_constructed,A_constructed_delB,AB_fro)
        err=norm(AB-(AB_constructed+A_constructed_delB),'fro')/AB_fro; 
end 
