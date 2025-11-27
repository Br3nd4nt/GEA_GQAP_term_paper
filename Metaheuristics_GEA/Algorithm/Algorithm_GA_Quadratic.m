function [Ans, BestCost, pop, time, Contribution_Rate]=Algorithm_GA_Quadratic(Solution, Info, Instraction)

global NFE;
NFE=0;

%% initialization
costfunction=@(q) Cost(q,Info.Model);       % Cost Function
NCrossover=2*round((Info.PCrossover*Info.Npop)/2);
NMutation=floor(Info.PMutation*Info.Npop);

NCrossover_Scenario=floor(Info.NCrossover_Scenario*(Info.PScenario3*Info.Npop));
NMutate_Scenario=floor(Info.NMutation_Scenario*(Info.PScenario3*Info.Npop));

BestCost=zeros(Info.Iteration,1);

individual.Position1=[];
individual.Xij=[];
individual.Cost=[];
individual.CVAR=[];
individual.delta = 0; % adaptive learning value

pop=repmat(individual,Info.Npop,1);
% popc=repmat(individual,NCrossover,1);
% popm=repmat(individual,NMutation,1);

% Adaptive learning parameters
% TODO: move into Info struct
epsilon = 1e-5; % small value for the case of dividing by zero
alpha = .01; % constant - rate of lambda growth
gamma = .5; % population selection
lambda_min = .4;
lambda_max = 1.5;

lambdas = [1, 1, 1, 1, 1];
delta_averages = [0, 0, 0, 0, 0];

if(Instraction(1))
    temp_Scens=repmat(individual,NCrossover_Scenario*2,1);
elseif(Instraction(2))
    temp_Scens=repmat(individual,NMutate_Scenario,1);
elseif(Instraction(3))
    temp_Scens=repmat(individual,NMutate_Scenario,1);
end

if (Instraction(1) && Instraction(2) && Instraction(3))
    temp_Scens=repmat(individual,(NCrossover_Scenario*2)+NMutate_Scenario+NMutate_Scenario, 1);
end


% BestSol=Solution;

[BestSol, ~] = find(Solution.Solution);
BestSol = reshape(BestSol, 1, size(BestSol,1));  % Convert the 2D solution to 1D chromosome
%% Initial population

pop(1).Position1 = BestSol;
[pop(1).Position1, pop(1).Xij] = CreateXij(pop(1).Position1, Info.Model);
[pop(1).Cost, pop(1).Xij, pop(1).CVAR]=CostFunction(pop(1).Xij, Info.Model);

for i=2:Info.Npop   

    pop(i).Position1 = Mutation(pop(1).Position1 ,Info.Model);
    % Create Xij
    [pop(i).Position1, pop(i).Xij] = CreateXij(pop(i).Position1, Info.Model);
    % Evaluation
    [pop(i).Cost, pop(i).Xij, pop(i).CVAR]=CostFunction(pop(i).Xij, Info.Model);

    while pop(i).Cost == inf
        pop(i).Position1 = Mutation(pop(1).Position1 ,Info.Model);
        % Create Xij
        [pop(i).Position1, pop(i).Xij] = CreateXij(pop(i).Position1, Info.Model);
        % Evaluation
        [pop(i).Cost, pop(i).Xij, pop(i).CVAR]=CostFunction(pop(i).Xij, Info.Model);
    end
end

% Sort Population
Costs=[pop.Cost];
[Costs, SortOrder]=sort(Costs);
pop=pop(SortOrder);
pop=pop(1:Info.Npop);

% Store Cost
BestSol=pop(1);
WorstCost=pop(end).Cost;
beta=10;         % Selection Pressure (Roulette Wheel)

tic;
%% GA Main loop
for It=1:Info.Iteration
    
    % Probability for Roulette Wheel Selection
    P=exp(-beta*Costs/WorstCost);
    P=P/sum(P);
    
    %% Crossover

    number_of_crossover_iterations = 2 * floor(NCrossover / 2 * lambdas(1));
    popc=repmat(individual,number_of_crossover_iterations,1);

    for k=1:2:number_of_crossover_iterations

        i1=RouletteWheelSelection(P);

        %TODO: Figure out Inf/-Inf situations

        while pop(i1).Cost == Inf || pop(i1).Cost == -Inf
            i1=RouletteWheelSelection(P);
        end


        i2=RouletteWheelSelection(P);

        while pop(i2).Cost == Inf || pop(i2).Cost == -Inf
            i2=RouletteWheelSelection(P);
        end

        pop_(1)=pop(i1);
        pop_(2)=pop(i2);

        better_parent_cost = min(pop_(1).Cost, pop_(2).Cost);

        [popc(k).Position1, popc(k+1).Position1]=Crossover(pop_, Info.Model);
        
        % Create Xij for new offspring
        [popc(k).Position1, popc(k).Xij] = CreateXij(popc(k).Position1, Info.Model);
        [popc(k+1).Position1, popc(k+1).Xij] = CreateXij(popc(k+1).Position1, Info.Model);

        % evaluate
        [popc(k).Cost, popc(k).Xij, popc(k).CVAR]=CostFunction(popc(k).Xij, Info.Model);

        [popc(k+1).Cost, popc(k+1).Xij, popc(k+1).CVAR]=CostFunction(popc(k+1).Xij, Info.Model);

        if popc(k, 1).Cost == Inf 
            popc(k, 1).delta = Inf;
        else 
            popc(k, 1).delta = (better_parent_cost - popc(k,1).Cost) / (better_parent_cost + epsilon);
        end

        if popc(k+1, 1).Cost == Inf 
            popc(k+1, 1).delta = Inf;
        else 
            popc(k+1, 1).delta = (better_parent_cost - popc(k+1,1).Cost) / (better_parent_cost + epsilon);
        end

        best_offspring_delta = min(popc(k, 1).delta, popc(k+1).delta);
        
        if best_offspring_delta == Inf || best_offspring_delta == -Inf
            best_offspring_delta = 0;
        end
        delta_averages(1) = delta_averages(1) + best_offspring_delta;
    end

    delta_averages(1) = delta_averages(1) / number_of_crossover_iterations;
    
    %% Mutation

    number_of_mutation_iterations = ceil(NMutation * lambdas(2));
    popm=repmat(individual,number_of_mutation_iterations,1);

    for k=1:number_of_mutation_iterations

        mutation_index = randsample(1:Info.Npop,1);

        popm(k).Position1=Mutation(pop(mutation_index).Position1 ,Info.Model);

        % Create Xij
        [popm(k).Position1, popm(k).Xij] = CreateXij(popm(k).Position1, Info.Model);

        % Evaluation
        [popm(k).Cost, popm(k).Xij, popm(k).CVAR]=CostFunction(popm(k).Xij, Info.Model);

        if popm(k).Cost == Inf || popm(k).Cost == -Inf || pop(mutation_index).Cost == Inf || pop(mutation_index).Cost == -Inf
            popm(k).delta = 0;
        else
            popm(k).delta = (pop(mutation_index).Cost - popm(k).Cost) / (pop(mutation_index).Cost + epsilon);
        end

        if popm(k).delta == Inf || popm(k).delta == -Inf
        else 
            delta_averages(2) = delta_averages(2) + popm(k).delta;
        end
    end

    delta_averages(2) = delta_averages(2) / number_of_mutation_iterations;
    

    %% scenario 1 : Dominated Gen
    if(Instraction(1))

        number_of_scenario_1_iterations = 2 * floor(NCrossover_Scenario * lambdas(3));

        [DominantGenes,Mask, DominantChromosome,Mask_Dominant]=Analyze_Perm(pop(1:(Info.PScenario1*Info.Npop)),Info);

        pop_sc1=repmat(individual,number_of_scenario_1_iterations,1);

        for k=1:2:number_of_scenario_1_iterations
            i1=RouletteWheelSelection(P);

            pop__(1)=DominantChromosome;
            pop__(2)=pop(i1);
            [pop_sc1(k).Position1, pop_sc1(k+1).Position1]=Crossover(pop__, Info.Model);

            % Create Xij for new offspring
            [pop_sc1(k).Position1, pop_sc1(k).Xij] = CreateXij(pop_sc1(k).Position1, Info.Model);
            [pop_sc1(k+1).Position1, pop_sc1(k+1).Xij] = CreateXij(pop_sc1(k+1).Position1, Info.Model);

            % evaluate
            [pop_sc1(k).Cost, pop_sc1(k).Xij, pop_sc1(k).CVAR]=CostFunction(pop_sc1(k).Xij, Info.Model);
            [pop_sc1(k+1).Cost, pop_sc1(k+1).Xij, pop_sc1(k+1).CVAR]=CostFunction(pop_sc1(k+1).Xij, Info.Model);

            better_parent_cost = min(pop__(1).Cost, pop__(2).Cost);

            if pop_sc1(k).Cost == Inf || pop_sc1(k).Cost == -Inf || ...
               better_parent_cost == Inf || better_parent_cost == -Inf
                pop_sc1(k).delta = 0;
            else
                pop_sc1(k).delta = (better_parent_cost - pop_sc1(k).Cost) / ...
                                   (better_parent_cost + epsilon);
            end
            
            if pop_sc1(k+1).Cost == Inf || pop_sc1(k+1).Cost == -Inf || ...
               better_parent_cost == Inf || better_parent_cost == -Inf
                pop_sc1(k+1).delta = 0;
            else
                pop_sc1(k+1).delta = (better_parent_cost - pop_sc1(k+1).Cost) / ...
                                     (better_parent_cost + epsilon);
            end
            
            best_offspring_delta = min(pop_sc1(k).delta, pop_sc1(k+1).delta);

            if best_offspring_delta == Inf || best_offspring_delta == -Inf
                best_offspring_delta = 0;
            end

            delta_averages(3) = delta_averages(3) + best_offspring_delta;
        end

        delta_averages(3) = delta_averages(3) / number_of_scenario_1_iterations;
    end

    % TODO: add the pop_sc1 to population!!!
    
    %% scenario 2 : mask mutation in goods
    if(Instraction(2))

        number_of_scenario_2_iterations = NMutate_Scenario*lambdas(4);

        [~,Mask,~,~]=Analyze_Perm(pop(1:(Info.PScenario2*Info.Npop)),Info);

        pop_sc2=repmat(individual,number_of_scenario_2_iterations,1);

        for i=1:number_of_scenario_2_iterations
            ii = randsample(1:(Info.PScenario2*Info.Npop),1);

            parent = pop(ii);

            pop_sc2(i).Position1 = MaskMutation(Info.MaskMutationIndex,parent.Position1,Mask(ii,:),Info.Model);
            
            % Create Xij for new offspring
            [pop_sc2(i).Position1, pop_sc2(i).Xij] = CreateXij(pop_sc2(i).Position1, Info.Model);

            % evaluate
            [pop_sc2(i).Cost, pop_sc2(i).Xij, pop_sc2(i).CVAR]=CostFunction(pop_sc2(i).Xij, Info.Model);

            if pop_sc2(i).Cost == Inf || pop_sc2(i).Cost == -Inf || ...
               parent.Cost == Inf || parent.Cost == -Inf
                pop_sc2(i).delta = 0;
            else
                pop_sc2(i).delta = (parent.Cost - pop_sc2(i).Cost) / ...
                                   (parent.Cost + epsilon);
            end

            if pop_sc2(i).delta == Inf || pop_sc2(i).delta == -Inf
            else 
                delta_averages(4) = delta_averages(4) + pop_sc2(i).delta;
            end

        end

        delta_averages(4) = delta_averages(4) / number_of_scenario_2_iterations;
    end
    
    %% scenario 3 : inject good gens
    if(Instraction(3))

        number_of_scenario_3_iterations = NMutate_Scenario * lambdas(5);

        [DominantGenes,Mask,~, Mask_Dominant]=Analyze_Perm(pop(1:(Info.PScenario3*Info.Npop)),Info);

        pop_sc3 = repmat(individual, number_of_scenario_3_iterations, 1);

        for z=1:number_of_scenario_3_iterations
            jj = randsample(size(pop,1)-(Info.PScenario3*Info.Npop):size(pop,1),1);

            parent = pop(jj);

            pop_sc3(z).Position1 = CombineQ(DominantGenes.Position1.Position1,parent.Position1,Mask_Dominant,Info.Model);
                        
            % Create Xij for new offspring
            [pop_sc3(z).Position1, pop_sc3(z).Xij] = CreateXij(pop_sc3(z).Position1, Info.Model);

            % evaluate
            [pop_sc3(z).Cost, pop_sc3(z).Xij, pop_sc3(z).CVAR]=CostFunction(pop_sc3(z).Xij, Info.Model);     

            if pop_sc3(z).Cost == Inf || pop_sc3(z).Cost == -Inf || ...
               parent.Cost == Inf || parent.Cost == -Inf
                pop_sc3(z).delta = 0;
            else
                pop_sc3(z).delta = (parent.Cost - pop_sc3(z).Cost) / ...
                                   (parent.Cost + epsilon);
            end

            if pop_sc3(z).delta == Inf || pop_sc3(z).delta == -Inf
            else 
                delta_averages(5) = delta_averages(5) + pop_sc3(z).delta;
            end
        end

        delta_averages(5) = delta_averages(5) / number_of_scenario_3_iterations;
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Calculate the contrubution share for Sensitivity Analysis GEA_3  %%%%%%%%%%%%%%
    index_pop = [1:Info.Npop];
    index_crossover = [Info.Npop+1:Info.Npop+numel(popc)];
    index_mutation = [Info.Npop+numel(popc)+1:Info.Npop+numel(popc)+numel(popm)];

    index_Scenario_3 = [Info.Npop+numel(popc)+numel(popm)+1:Info.Npop+numel(popc)+numel(popm)+numel(temp_Scens)];
    
    %% Pool fusion & Selection Best Chromosome

    % Elitism Selection (Npop best will be selected)
    % Create Merged Population
    if (size(temp_Scens,1)>1)
        pop=[pop;popc;popm;temp_Scens];
    else
        pop=[pop;popc;popm];
    end

    Costs=[pop.Cost];
    [Costs, SortOrder]=sort(Costs);
    pop=pop(SortOrder); 
    pop=pop(1:Info.Npop);
    Costs=[pop.Cost];
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Calculate share rate for all operators for GEA_3 %%%%%%%%%%%%%%%%%%%%%%%%%
    previous_pop = size(intersect(SortOrder(1:Info.Npop), index_pop), 2) / Info.Npop;
    crossover_ShareRate = size(intersect(SortOrder(1:Info.Npop), index_crossover), 2) / Info.Npop;
    mutation_ShareRate = size(intersect(SortOrder(1:Info.Npop), index_mutation), 2) / Info.Npop;
    Scenario3_ShareRate = size(intersect(SortOrder(1:Info.Npop), index_Scenario_3), 2) / Info.Npop;
    Contribution_Rate(It, :) = [previous_pop crossover_ShareRate mutation_ShareRate Scenario3_ShareRate];

    % Update Worst Cost
    WorstCost=max(WorstCost,pop(end).Cost);
    
    % Store Best Solution Ever Found
    if pop(1).Cost < BestSol.Cost
    BestSol=pop(1);
    end

    BestCost(It)=BestSol.Cost;
    BestPosition=pop(1).Position1;
    
    % Store NFE
    nfe(It)=NFE;
    
   % Show Iteration Information
    disp(['Iteration ' num2str(It)  ', Best Cost = ' num2str(BestCost(It))]);
    disp(delta_averages)
    time = toc;
    if time>=1000
        break;
    end
    
end
time;    
Ans=pop(1).Cost;
Solution=BestSol; 
end