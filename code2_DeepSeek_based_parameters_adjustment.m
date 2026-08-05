
function [pop, pcrossover_set_LLM, pmutation_set_LLM, time_LLM_pure] = stage_2_NSGA_II_DeepSeek(location_origin_uav_transpose, location_users, ...
                                          location_server, pop, num_uav, num_user, ...
                                          mat_sem_table, mat_sentences, lb, ub, empty_individual, ...
                                          pcrossover_set_LLM, pmutation_set_LLM, ...
                                          time_LLM_pure, ext_it, MaxIt_local) 

    import matlab.net.*
    import matlab.net.http.*

    %% Parameters
    MaxIt_local = 10;                         % Local Iterations
    nPop = numel(pop);                        % Number of Pop
    
    pCrossover = 0.8;                         % Crossover Percentage
    pMutation = 0.4;                          % Mutation Percentage

    nCrossover = round(pCrossover*nPop);      % Number of Parents (Offsprings)
    nMutation = round(pMutation*nPop);        % Number of Mutants
    
    mu = 0.02;                                % Mutation Rate，
    
    sigma = 0.1*(ub - lb);                    % Mutation step 
    
    id_xyzi = num_uav*1 + 1:num_uav*5;

    prev_SP = NaN;
    prev_M3_star = NaN;

    %% Content of deepseek
    % Define the API information of deepseek
    api_key = "xxxxxxxxxx";
    api_url = "https://api.deepseek.com/v1/chat/completions";

    %% Update the current particle

    for it = 1:MaxIt_local
        
        tic;

        % Crossover
        popc = repmat(empty_individual, floor(nCrossover/2), 1); 
        for k = 1:nCrossover/2
            
            i1 = randi([1 numel(pop)]);
            p1 = pop(i1);
            
            i2 = randi([1 numel(pop)]);
            p2 = pop(i2);

            [p1.Position(1, id_xyzi), p2.Position(1, id_xyzi)] = Crossover(p1.Position(1, id_xyzi), p2.Position(1, id_xyzi));

            popc(k, 1).Position = p1.Position;
            popc(k, 2).Position = p2.Position;
          
            popc(k, 1).Cost = objective_function_v2(location_origin_uav_transpose, location_users, location_server, ...
                      popc(k, 1).Position, num_uav, num_user, mat_sem_table, mat_sentences);   
            popc(k, 2).Cost = objective_function_v2(location_origin_uav_transpose, location_users, location_server, ...
                       popc(k, 2).Position, num_uav, num_user, mat_sem_table, mat_sentences);
        end
        popc = popc(:);

        % Mutation
        popm = repmat(empty_individual, nMutation, 1);
        for k = 1:nMutation
           
            i = randi([1 numel(pop)]);
            p = pop(i);

            popm(k).Position = p.Position;

            popm(k).Position(1, id_xyzi) = Mutate(p.Position(1, id_xyzi), mu, sigma(1, id_xyzi));    
            popm(k).Cost = objective_function_v2(location_origin_uav_transpose, location_users, location_server, ...
                       popm(k).Position, num_uav, num_user, mat_sem_table, mat_sentences);
        end
        
        % Merge
        pop = [pop 
               popc
               popm]; 

        % Non-Dominated Sorting
        [pop, F] = NonDominatedSorting(pop);
    
        % Calculate Crowding Distance
        pop = CalcCrowdingDistance(pop, F);
    
        % Sort Population
        pop = SortPopulation(pop);
        
        % Truncate
        if numel(pop)>nPop   
                pop = pop(1:nPop);
        end
        
        % Non-Dominated Sorting
        [pop, F] = NonDominatedSorting(pop);
    
        % Calculate Crowding Distance
        pop = CalcCrowdingDistance(pop, F);
    
        % Sort Population
        [pop, F] = SortPopulation(pop);

        % Calculate metrixs: SP and M3*
        [SP, M3_star] = calculate_sp_m3star(pop);

        % Adjust Parameters
        template = [ ...
            'You are an intelligent optimization assistant for an NSGA-II algorithm. Your objective is to adaptively update the crossover probability and the mutation probability based on the following parameters, guidance, hard constraints, and output rules.\n', ...
            '[Current Iteration Parameters:]\n', ...
            'Current local iteration: %d\n', ...
            'Maximum number of local iteration: %d\n', ...
            'Current crossover probability: %g\n', ...
            'Current mutation probability: %g\n', ...
            'SP: %g (spacing metric, smaller value indicates better distribution uniformity)\n', ...
            'M3*: %g (extension metric, larger value indicates better spread)\n', ...
            'Previous SP: %g\n', ...
            'Previous M3*: %g\n', ...
            'Population size: %d\n', ...
            '[Optimization Guidance and Policy:]\n', ...
            '1. SP reflects the distribution uniformity of the current Pareto front. Larger values generally indicate poorer uniformity.\n', ...
            '2. M3* reflects the diversity spread of the Pareto front. Larger values generally indicate better diversity.\n', ...
            '3. The adjustment of crossover and mutation probabilities should balance exploration and exploitation according to the current optimization status.\n', ...
            '4. Based on the comparison between Previous SP and Current SP, and between Previous M3* and Current M3*, adjust the probabilities to improve convergence and diversity. If the current SP becomes larger than the previous SP, slightly increase mutation or reduce crossover; if the current M3* becomes smaller than the previous M3*, moderately increase crossover.\n', ...
            '5. When the optimization exhibits stagnation, stronger exploration may be considered.\n', ...
            '6. As the optimization proceeds, the search should gradually emphasize convergence while maintaining sufficient diversity.\n', ...
            '[Hard Constraints - DO NOT VIOLATE:]\n', ...
            '1. After adjustment, you MUST clip both values into [0, 1].\n', ...
            '2. Clipping must be done internally and silently; the final output must be numbers only.\n', ...
            '3. You are NOT allowed to output expressions like max(), min(), clip(), parentheses, or formulas.\n', ...
            '4. You MUST output only the final numeric values after clipping.\n', ...
            '[Output Rules:]\n', ...
            '1. Output ONLY one line of MATLAB code:\n', ...
            '   pCrossover_updated = x, pMutation_updated = y.\n', ...
            '2. x and y must be the final numeric values after clipping.\n' ...
        ];
        
        prompt_text = sprintf(template, it, MaxIt_local, pCrossover, pMutation, SP, M3_star, prev_SP, prev_M3_star, nPop);
        
        % Prepare request body (chat API format)
        body = struct( ...
            'model', "deepseek-v4-pro", ...
            'messages', { ...
                {struct('role', "user", 'content', prompt_text)} ...
            } ...
        );
        
        % Headers
        headers = [ ...
            HeaderField("Content-Type", "application/json"), ...
            HeaderField("Authorization", "Bearer " + api_key) ...
        ];
        
        % Send HTTP request
        old_pCrossover = pCrossover;
        old_pMutation = pMutation;
        request = RequestMessage('post', headers, body);
        t_llm_start = tic;
        response_text = '';
        try
            response = request.send(api_url);
            time_LLM_pure(ext_it, it) = toc(t_llm_start);

            %% ========== 5. 解析返回内容 ==========
            if response.StatusCode == 200
                data = response.Body.Data;   % <-- 正确解析方式
                disp("=== DeepSeek 回复： ===");
                disp(data.choices(1).message.content);
                response_text = data.choices(1).message.content;
            else
                disp("请求失败：");
                disp(response.StatusCode);
                disp(response.Body.Data);
            end
        catch ME
            warning('LLM request failed: %s', ME.message);
        end

        % 1. 使用正则表达式匹配数字串
        % \d+ 匹配一个或多个数字
        pattern = 'pCrossover_updated\s*=\s*(?<pc>[0-9]*\.?[0-9]+).*pMutation_updated\s*=\s*(?<pm>[0-9]*\.?[0-9]+)';

        % 执行正则匹配
        match = regexp(response_text, pattern, 'names');
        
        if ~isempty(match)
            % 转换为数字
            new_pC = str2double(match.pc);
            new_pM = str2double(match.pm);
            
            % 容错检查：防止 str2double 返回 NaN
            if ~isnan(new_pC) && ~isnan(new_pM)
                pCrossover = new_pC;
                pMutation = new_pM;
                fprintf('成功提取！pCrossover: %.4f, pMutation: %.4f\n', pCrossover, pMutation);
            else
                disp('数值转换失败 (包含非数字字符)。保留上一次参数。');
                pCrossover = old_pCrossover;
                pMutation = old_pMutation;
            end
        else
            disp('未能按预期格式找到参数，请检查回复。保留上一次参数。');
            pCrossover = old_pCrossover;
            pMutation = old_pMutation;
        end
        
        pcrossover_set_LLM = [
            pcrossover_set_LLM
            pCrossover];

        pmutation_set_LLM = [
            pmutation_set_LLM
            pMutation];
        
        pCrossover = min(max(pCrossover, 0), 1);
        pMutation = min(max(pMutation, 0), 1);

        prev_SP = SP;
        prev_M3_star = M3_star;

        nCrossover = round(pCrossover*nPop);      % Number of Parents (Offsprings)
        nMutation = round(pMutation*nPop);        % Number of Mutants

        disp(['In Stage 2, pop updated in internal iteration ' num2str(it)]);
    end
    
end
