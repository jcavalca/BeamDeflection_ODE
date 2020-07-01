clc; clear; close all;
%% Joao Cavalcanti - Problem 1

E = 210e9; 
I = 1e-8/12;
L = 1;

[v_4, x_domain] = vcalc(16);
actual_v = @(x) (1000/(24*E*I))*x.^2.*(x-L).^2;

figure(1)

plot(x_domain, v_4)
hold on
plot(x_domain, actual_v(x_domain))
title("n = 16")
legend("Estimate", "Real")

% n = 2^4

norm_4 = infNorm(v_4, actual_v(x_domain));


% n = 32
E = 210e9; 
I = 1e-8/12;

[v_5, x_domain] = vcalc(32);
norm_5 = infNorm(v_5, actual_v(x_domain));

% n = 64

[v_6, x_domain] = vcalc(2^6);
norm_6 = infNorm(v_6, actual_v(x_domain));

% n = 128

[v_7, x_domain] = vcalc(2^7);
norm_7 = infNorm(v_7, actual_v(x_domain));

% n = 256

[v_8, x_domain] = vcalc(2^8);
norm_8 = infNorm(v_8, actual_v(x_domain));
% n = 512
[v_9, x_domain] = vcalc(2^9);
norm_9 = infNorm(v_9, actual_v(x_domain));

% n = 1028
[v_10, x_domain] = vcalc(2^10);
norm_10 = infNorm(v_10, actual_v(x_domain));


n_vector = [16, 32, 64, 128, 256, 512, 1024];
norm_vector = [norm_4, norm_5, norm_6, norm_7, norm_8, norm_9, norm_10];

res = polyfit(log(n_vector), log(norm_vector), 1);
slope = res(1);

figure(2)
loglog(n_vector, norm_vector)
title("Log Plot of N and Infinity Norm")

disp("Order of P is " + abs(slope) + " (around 2 0(h^2) ), which agrees with was expected previously")


%% Calculating V Vector for an Arbitrary n

function [v, x_domain] = vcalc(n)

E = 210e9; 
I = 1e-8/12;
L = 1;
h = L/n;

q(1:1:n-1, 1) = (1000*h^4)/(E*I);

A = matrix_A(n - 1);
 
old_v = A\q;

x_domain = linspace(0, 1, n+1);

v = zeros(n+1, 1);
v(1) = 0;
v(n+1) = 0;
v(2:1:n) = old_v(1:1:n-1);


end



%% Infinity-Norm Function For an Arbitrary Vector

function [inf_norm] = infNorm(v, actual_v)

n = size(v) - 1;

max = abs(v(1) - actual_v(1));

for i = 2:1:n+1
    current = abs(v(i) - actual_v(i));
    if current > max
        max = current;
    end
end

inf_norm = max;

end


%% A Function -> Creating A matrix for an arbitrary n

function [A] = matrix_A(size)

A = zeros(size, size);

for i = 1:1:size
    A(i, i) = 7;
    if i == 1
        A(1, 1) = 7;
        A(1, 2) = -4;
        A(1, 3) = 1;
    elseif i == size 
        A(i, i) = 7;
        A(i, i - 1) = -4;
        A(i, i - 2) = 1;
    elseif i == 2
        A(i, i-1) = -4;
        A(i, i) = 6;
        A(i, i +1) = -4; 
        A(i, i+2) = 1;
    elseif i == size -1
        A(i, i-2) = 1;
        A(i, i-1) = -4;
        A(i, i) = 6;
        A(i, i +1)= -4;
    else
        A(i, i - 2) = 1;
        A(i, i - 1) = -4;
        A(i, i) = 6;
        A(i, i + 1) = -4;
        A(i, i + 2) =  1;
    end 
end

end



    