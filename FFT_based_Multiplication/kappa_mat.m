function [matrix]= kappa_mat(n)
   
    matrix = zeros(n,n);

    
    for i =0:n-1
        for j =i:n-1  
            
            abs_diff = abs(i - j);
            
           
            exp_term = exp(-0.5 * abs_diff);
            
           
            sin_term = sin(i + 1);
            
            
            value = exp_term * sin_term;
            
            
            matrix(i+1, j+1) = value;
            matrix(j+1, i+1) = value;

        end
    end
end