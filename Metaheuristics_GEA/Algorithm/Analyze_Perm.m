function [DominantGenes, Mask, DominantChromosome, Mask_Dominant]=Analyze_Perm(pop,Info)
%% Data Definition
Npop=size(pop,1);
% n=size(pop(1).Position1,2);
% NFixedX=floor(Info.PFixedX*Npop);
% Mask=zeros(Info.PScenario1*Info.Npop, n);
%% Find Dominant Gene
Mask = vectorized_mask_creation(pop, floor(Info.PFixedX * Info.PScenario1 * Npop));

%% Create Ans :
count = 0;
Domin = [];
Mask_Dominant = [];

delta = 0;

%% for i=1:Npop
for i=1:length(Mask)
    temp=sum(Mask(i,:)==1);
    % Mask(i,:);
    if (temp>=count)
        if (size(Domin,2)==0)
            Domin=pop(i).Position1;
            Mask_Dominant=Mask(i,:);
        else
            decision = rand(1);
            if (decision>0.5)
                Domin=pop(i).Position1;
                delta = pop(i).delta;
                Mask_Dominant=Mask(i,:);
            end
        end
        count = temp;
    end
end

DominantChromosome.Position1=Domin;
[DominantChromosome(1).Position1, DominantChromosome(1).Xij] = CreateXij(DominantChromosome(1).Position1, Info.Model);
% evaluate
[DominantChromosome(1).Cost, DominantChromosome(1).Xij, DominantChromosome(1).CVAR]=CostFunction(DominantChromosome(1).Xij, Info.Model);
DominantGenes.Position1 = DominantChromosome;

DominantChromosome.delta = delta;


end
