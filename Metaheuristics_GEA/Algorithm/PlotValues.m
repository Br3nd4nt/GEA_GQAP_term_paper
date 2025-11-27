function [] = PlotValues(lambda_history,delta_history)
    opNames = {'Crossover','Mutation','Scenario 1','Scenario 2','Scenario 3'};
    
    % ---- Plot Lambda History ----
    figure('Name','Lambda History','NumberTitle','off');
    plot(lambda_history, 'LineWidth', 1.5);
    grid on;
    xlabel('Iteration');
    ylabel('\lambda');
    title('\lambda history for all operations');
    legend(opNames, 'Location','best');
    set(gca,'FontSize',12);
    
    
    % ---- Plot Delta History ----
    figure('Name','Delta History','NumberTitle','off');
    plot(delta_history, 'LineWidth', 1.5);
    grid on;
    xlabel('Iteration');
    ylabel('\delta');
    title('\delta history for all operations');
    legend(opNames, 'Location','best');
    set(gca,'FontSize',12);
    end

