clc; clear; close all;

% fixing range of sensors, calculating communication range
area_length = 100; % in meters => area = 100 x 100 sqm
n = 20:5:200; % number of (potential) sensors deployed in RGG
r = zeros(size(n));

for i = 1:length(n)
    r(i) = sqrt(2 * log(n(i)) / (n(i))) * area_length;
end

figure;
plot(n, r, '-o', LineWidth=2);
xlabel("Number of sensors deployed");
ylabel("Communication radius (in meters)");
title("Tradeoff in sensor deployment");
grid on;

% let us now assume a constant radius, say 50 m
comm_radius = 20;
num_trials = 200; % number of Monte Carlo trials
% min_sensors = inf; % worst-case (unrealistic) scenario

connectivity_prob = zeros(size(n));

for k = 1:length(n)
    success = 0;

    for t = 1:num_trials
        % random positions
        % rng(42);
        pos = area_length * rand(n(k), 2); % uniformly deployed

        % adjacency matrix
        A = zeros(n(k));
        for i = 1:n(k)
            for j = i+1:n(k)
                if norm(pos(i,:) - pos(j,:)) <= comm_radius
                    A(i,j) = 1;
                    A(j,i) = 1;
                end
            end
        end

        % graph connectivity
        G = graph(A);
        if all(conncomp(G) == 1)
            success = success + 1;
        end
    end

    % probability of connectivity
    connectivity_prob(k) = success / num_trials;
end

figure;
plot(n, connectivity_prob, '-o', LineWidth=2);
xlabel('Number of sensors deployed');
ylabel('Connectivity probability');
title(sprintf('Connectivity vs. Sensors (r = %d, trials = %d)', comm_radius, num_trials));
grid on;

% from above two graphs, taking comm_radius = 20, num_sensors = 100
% increasing sensors by 5 to show good convergence in median consensus
% using same graph
% because there is a very high probability of connectivity

num_sensors = 100;
% num_sensors = 105;
sensor_positions = area_length * rand(num_sensors, 2);

% adjacency matrix
adjacency_matrix = zeros(num_sensors);
for i = 1:num_sensors
    for j = i+1:num_sensors
        if norm(sensor_positions(i,:) - sensor_positions(j,:)) <= comm_radius
            adjacency_matrix(i,j) = 1;
            adjacency_matrix(j,i) = 1;
        end
    end
end

% graph (sensor network) using adjacency matrix
sensor_network = graph(adjacency_matrix);

if all(conncomp(sensor_network) == 1)
    disp("Connected graph formed!");
    figure;
    scatter(sensor_positions(:, 1), sensor_positions(:, 2), 100, 'filled');
    xlabel('Positions along x-axis');
    ylabel('Positions along y-axis');
    grid on;
    hold on;

    % communication links
    for i = 1:num_sensors
        for j = i+1:num_sensors
            if adjacency_matrix(i, j) == 1
                plot([sensor_positions(i, 1), sensor_positions(j, 1)], ...
                    [sensor_positions(i, 2), sensor_positions(j, 2)], 'b--', 'LineWidth', 0.5);
            end
        end
    end

    title(sprintf('Random Geometric Graph (r = %d, n = %d)', comm_radius, num_sensors));
end

%% parameters per sensor
actual_value = 30;
reading_error = 2; % reading error
x_initial = reading_error .* randn(num_sensors, 1) + actual_value; % initial values
true_avg = mean(x_initial);
max_iterations = 200000; % safety cap on iterations
tolerance = 1e-6; % convergence tolerance

%% randomized gossip implementation (average consensus)
x_gossip = x_initial;
% x_gossip_history = zeros(max_iterations, num_sensors);
error_history_gossip = zeros(max_iterations, 1);
transmissions_gossip = 0;
convergence_time_gossip = 0;

for iter_gossip = 1:max_iterations
    error_history_gossip(iter_gossip) = norm(x_gossip - true_avg);

    % convergence check
    if iter_gossip > 1 && norm(x_gossip - true_avg) < tolerance
        convergence_time_gossip = iter_gossip;
        error_history_gossip = error_history_gossip(1:iter_gossip, :);
        break;
    end

    if iter_gossip > 1 && norm(x_gossip - true_avg) > 1e3
        fprintf("Graph diverging!!!");
        convergence_time_gossip = iter_gossip;
        error_history_gossip = error_history_gossip(1:iter_gossip, :);
        break;
    end

    % randomized gossip step
    % selecting random node
    i = randi(num_sensors);

    % finding neighbors of node i
    neighbors_i = find(adjacency_matrix(i,:));

    if ~isempty(neighbors_i)
        % selecting random neighbor
        j = neighbors_i(randi(length(neighbors_i)));

        % gossip update
        avg_value = (x_gossip(i) + x_gossip(j)) / 2;
        x_gossip(i) = avg_value;
        x_gossip(j) = avg_value;

        transmissions_gossip = transmissions_gossip + 2; % 2 transmissions per exchange
    end
end

if iter_gossip == max_iterations
    convergence_time_gossip = iter_gossip;
    fprintf("WARNING: Maximum iterations reached without convergence!\n");
end

trans_per_iter = 2;
transmission_axis_gossip = trans_per_iter * (1:length(error_history_gossip));

% plotting convergence
figure;
semilogy(transmission_axis_gossip, error_history_gossip);
title('Randomized Gossip Convergence (Average Consensus)');
xlabel('Transmissions'); ylabel('MSE w.r.t. True Average');
grid on;

fprintf('Gossip (Average Consensus) - Convergence time: %d iterations, Transmissions: %d\n', ...
            convergence_time_gossip, transmissions_gossip);

% [i_list, j_list] = find(triu(adjacency_matrix)); % indices of upper triangle of adjacency matrix
% 
% x_gossip = x_initial;
% transmissions_rg = 0;
% error_history_rg = zeros(max_iterations, 1);
% 
% for iter = 1:max_iterations
%     % randomly choosing a connected edge
%     idx = randi(length(i_list));
%     i = i_list(idx);
%     j = j_list(idx);
% 
%     % updating values
%     avg = (x_gossip(i) + x_gossip(j)) / 2;
%     x_gossip(i) = avg;
%     x_gossip(j) = avg;
% 
%     % tracking number of transmissions (2 per exchange)
%     transmissions_rg = transmissions_rg + 2;
% 
%     % tracking error
%     % error_history_rg(iter) = max(abs(temp_readings_gossip - true_avg));
%     error_history_rg(iter) = norm(x_gossip - true_avg) / sqrt(num_sensors);
% 
%     % convergence check
%     if error_history_rg(iter) < epsilon
%         break;
%     end
% end
% 
% fprintf("Randomized Gossip for Average Consensus converged in %d iterations and %d transmissions\n", iter, transmissions_rg);
% 
% % plotting convergence over time
% figure;
% semilogy(error_history_rg, LineWidth=2);
% axis([0 ceil(iter / 1000)*1000 1e-3 1]);
% xlabel('Iteration');
% % ylabel('Max. Error from True Average');
% ylabel('RMSE w.r.t. True Average');
% title('Convergence of Randomized Gossip (Average Consensus)');
% grid on;

%% PDMM implementation (average consensus)
x_pdmm_avg = x_initial;
degree_per_node = sum(adjacency_matrix, 2); % degree of each node

% auxiliary variables
z_pdmm_avg = zeros(num_sensors); % z(i,j) = z_{i|j}
y_pdmm_avg = zeros(num_sensors); % temporary - y_{i|j}
c = 1 / max(degree_per_node); % hyperparameter

error_history_pdmm_avg = zeros(max_iterations, 1);
convergence_time_pdmm_avg = 0;

% A-weights per edge pair - stored using adjacency matrix structure
A = adjacency_matrix;
A(triu(true(size(A)), 1)) = -A(triu(true(size(A)), 1));

for iter_pdmm_avg = 1:max_iterations
    error_history_pdmm_avg(iter_pdmm_avg) = norm(x_pdmm_avg - true_avg);

    % convergence check
    if iter_pdmm_avg > 1 && norm(x_pdmm_avg - true_avg) < tolerance
        convergence_time_pdmm_avg = iter_pdmm_avg;
        error_history_pdmm_avg = error_history_pdmm_avg(1:iter_pdmm_avg, :);
        break;
    end

    if iter_pdmm_avg > 1 && norm(x_pdmm_avg - true_avg) > 1e3
        fprintf("Graph diverging!!!");
        convergence_time_pdmm_avg = iter_pdmm_avg;
        error_history_pdmm_avg = error_history_pdmm_avg(1:iter_pdmm_avg, :);
        break;
    end

    % primal update
    x_new = zeros(num_sensors, 1);
    for i = 1:num_sensors
        sum_z = 0;
        for j = find(adjacency_matrix(i,:))
            sum_z = sum_z + A(i,j) * z_pdmm_avg(i,j);
        end
        x_new(i) = (x_initial(i) - sum_z) / (1 + c * degree_per_node(i));
    end

    % y update - y_{i|j} = z_{i|j} + 2c * A_{ij} * x_i
    for i = 1:num_sensors
        for j = find(adjacency_matrix(i,:))
            y_pdmm_avg(i,j) = z_pdmm_avg(i,j) + 2 * c * A(i,j) * x_new(i);
        end
    end

    % z update - z_{j|i} = y_{i|j}
    for i = 1:num_sensors
        for j = find(adjacency_matrix(i,:))
            z_pdmm_avg(i,j) = y_pdmm_avg(j,i);
        end
    end

    x_pdmm_avg = x_new;
end

if iter_pdmm_avg == max_iterations
    convergence_time_pdmm_avg = iter_pdmm_avg;
    fprintf("WARNING: Maximum iterations reached without convergence!\n");
end

transmissions_per_iter = 2 * sum(degree_per_node);
transmissions_axis_pdmm_avg = transmissions_per_iter * (1:length(error_history_pdmm_avg));
transmissions_pdmm_avg = (convergence_time_pdmm_avg - 1) * transmissions_per_iter;

% plotting convergence
figure;
semilogy(transmissions_axis_pdmm_avg, error_history_pdmm_avg);
title('PDMM Convergence (Average Consensus)');
xlabel('Transmissions'); ylabel('MSE w.r.t. True Average');
grid on;

fprintf('PDMM (Average Consensus) - Convergence time: %d iterations, Transmissions: %d\n', ...
            convergence_time_pdmm_avg, transmissions_pdmm_avg);

% convergence comparison with randomized gossip
figure;
semilogy(transmission_axis_gossip, error_history_gossip);
hold on;
semilogy(transmissions_axis_pdmm_avg, error_history_pdmm_avg);
title(sprintf('Average Consensus on RGG - Convergence Comparison(r = %d, n = %d)', comm_radius, num_sensors));
xlabel('Transmissions'); ylabel('MSE w.r.t. True Average');
legend('Randomized Gossip', 'PDMM');
grid on;

%% PDMM implementation (median consensus)
true_median = median(x_initial);
x_pdmm_med = x_initial;
tolerance_med = 1e-9;
max_iterations_median = max_iterations / 50;

error_history_pdmm_med = zeros(max_iterations_median, 1);
convergence_time_pdmm_med = 0;

% auxiliary variables
z_pdmm_med = zeros(num_sensors); % z(i,j) = z_{i|j}
y_pdmm_med = zeros(num_sensors); % temporary - y_{i|j}

for iter_pdmm_med = 1:max_iterations_median
    error_history_pdmm_med(iter_pdmm_med) = norm(x_pdmm_med - true_median);

    % convergence check
    if iter_pdmm_med > 1 && norm(x_pdmm_med - true_median) < tolerance_med
        convergence_time_pdmm_med = iter_pdmm_med;
        error_history_pdmm_med = error_history_pdmm_med(1:iter_pdmm_med, :);
        break;
    end

    if iter_pdmm_med > 1 && norm(x_pdmm_med - true_median) > 1e3
        fprintf("Graph diverging!!!");
        convergence_time_pdmm_med = iter_pdmm_med;
        error_history_pdmm_med = error_history_pdmm_med(1:iter_pdmm_med, :);
        break;
    end

    % primal update
    x_new = zeros(num_sensors, 1);
    for i = 1:num_sensors
        sum_z = 0;
        for j = find(adjacency_matrix(i,:))
            sum_z = sum_z + A(i,j) * z_pdmm_med(i,j);
        end
        if (-1 - sum_z) / (c * degree_per_node(i)) > x_initial(i)
            x_new(i) = (-1 - sum_z) / (c * degree_per_node(i));
        elseif (1 - sum_z) / (c * degree_per_node(i)) < x_initial(i)
            x_new(i) = (1 - sum_z) / (c * degree_per_node(i));
        else
            x_new(i) = x_initial(i);
        end
    end

    % y update - y_{i|j} = z_{i|j} + 2c * A_{ij} * x_i
    for i = 1:num_sensors
        for j = find(adjacency_matrix(i,:))
            y_pdmm_med(i,j) = z_pdmm_med(i,j) + 2 * c * A(i,j) * x_new(i);
        end
    end

    % z update - z_{i|j} = 0.5z_{i|j} + 0.5y_{j|i}
    for i = 1:num_sensors
        for j = find(adjacency_matrix(i,:))
            z_pdmm_med(i,j) = 0.5 * z_pdmm_med(i,j) + 0.5 * y_pdmm_med(j,i);
            % z_pdmm_med(i,j) = y_pdmm_med(j,i);
        end
    end

    x_pdmm_med = x_new;
end

if iter_pdmm_med == max_iterations_median
    convergence_time_pdmm_med = iter_pdmm_med;
    fprintf("WARNING: Maximum iterations reached without convergence!\n");
end

transmissions_axis_pdmm_med = transmissions_per_iter * (1:length(error_history_pdmm_med));
transmissions_pdmm_med = (convergence_time_pdmm_med - 1) * transmissions_per_iter;

% plotting convergence
figure;
semilogy(transmissions_axis_pdmm_med, error_history_pdmm_med);
title('PDMM Convergence (Median Consensus)');
xlabel('Transmissions'); ylabel('MSE w.r.t. True Median');
grid on;

fprintf('PDMM (Median Consensus) - Convergence time: %d iterations, Transmissions: %d\n', ...
            convergence_time_pdmm_med, transmissions_pdmm_med);