%% Stage 2
function [pop, pcrossover_set_LLM, pmutation_set_LLM] = code1_chatgpt_based_parameters_adjustment(location_origin_uav_transpose, location_users, ...
                                          location_server, pop, num_uav, num_user, ...
                                          mat_sem_table, mat_sentences, lb, ub, empty_individual, ...
                                          pcrossover_set_LLM, pmutation_set_LLM)  

    import matlab.net.*
    import matlab.net.http.*

    %% Parameters
    MaxIt_local = 10;                         % Local Iterations
    nPop = numel(pop);                        % Number of Pop
    
    % 这里需要修改一下，如果当前迭代是1，表示第1次迭代，那么这个概率为自己设置的值；
    % 如果当前迭代不是1，那么这个概率就是从数组中取值
    pCrossover = 0.8;                         % Crossover Percentage
    pMutation = 0.4;                          % Mutation Percentage

    nCrossover = round(pCrossover*nPop);      % Number of Parents (Offsprings)
    nMutation = round(pMutation*nPop);        % Number of Mutants
    
    mu = 0.02;                                % Mutation Rate，
    
    sigma = 0.1*(ub - lb);                    % Mutation step 
    
    id_xyzi = num_uav*1 + 1:num_uav*5;

    %% Content of LLM
    % Define the API endpoint Davinci
    api_endpoint = "https://api.openai.com/v1/chat/completions";
    
    % Define the API key from https://beta.openai.com/account/api-keys
    api_key = "xxxxxx";

    %% Update the current particle

    for it = 1:MaxIt_local
        
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
        % 预处理提示词模板（使用 %g 以兼容浮点数和整数）
        template = [ ...
            'You are an intelligent optimization assistant for an NSGA-II algorithm. ', ...
            'Your task is to adaptively update the crossover probability (pCrossover) and mutation probability (pMutation). \n', ...
            '[Current Iteration Parameters:] \n', ...
            '- pCrossover: %g \n', ...
            '- pMutation: %g \n', ...
            '- SP: %g (spacing metric, smaller = better distribution uniformity) \n', ...
            '- M3*: %g (extension metric, larger = better spread) \n', ...
            '- Population size: %d \n', ...
            '- Number of objectives: 3 \n', ...
            '- Dimension of decision variables: %d \n\n', ...
            '[Optimization Targets:] \n', ...
            '1. If SP is large, increase mutation slightly and/or reduce crossover. \n', ...
            '2. If M3* is small, increase crossover slightly. \n', ...
            '[Hard Constraints - DO NOT VIOLATE:] \n', ...
            '1. After adjustment, you MUST clip both values into [0.05, 0.95]. \n', ...
            '2. Clipping must be done internally and silently; the final output must be numbers only. \n', ...
            '3. You are NOT allowed to output expressions like max(), min(), clip(), parentheses, or formulas. \n', ...
            '4. You MUST output only the final numeric values after clipping.\n\n', ...
            '[Output Rules] \n', ...
            'Output ONLY one line of MATLAB code: \n', ...
            'pCrossover_updated = x, pMutation_updated = y.\n', ...
            'where x and y are the values after being clipped.\n' ...
        ];
        
        % 执行格式化
        prompt_text = sprintf(template, pCrossover, pMutation, SP, M3_star, nPop, size(lb, 2));
        
        % Prepare request body (chat API format)
        body = struct( ...
            'model', "gpt-4o-mini", ...  % 推荐使用这个模型
            'messages', { ...
                {struct('role', "user", 'content', prompt_text)} ...
            } ...
        );
        
        % Headers
        headers = [ ...
            HeaderField("Content-Type", "application/json"), ...
            HeaderField("Authorization", "Bearer " + api_key) ...
        ];
        
        % ... (前面的代码不变) ...
        
        % Send HTTP request
        request = RequestMessage('post', headers, body);
        response = request.send(api_endpoint);
        
        % Extract response text
        response_data = response.Body.Data;
        
        if isfield(response_data, 'choices')
            response_text = response_data.choices(1).message.content; 
            disp(response.Body.Data);
            disp(response_text);
        else
            disp('Error: Could not find "choices" field. Full response data:');
            disp(response.Body.Data);
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
                disp('数值转换失败 (包含非数字字符)。');
            end
        else
            disp('未能按预期格式找到参数，请检查回复。');
        end
        
        pcrossover_set_LLM = [
            pcrossover_set_LLM
            pCrossover];

        pmutation_set_LLM = [
            pmutation_set_LLM
            pMutation];
        
        
        pCrossover = min(max(pCrossover, 0), 1);
        pMutation = min(max(pMutation, 0), 1);

        nCrossover = round(pCrossover*nPop);      % Number of Parents (Offsprings)
        nMutation = round(pMutation*nPop);        % Number of Mutants

        disp(['In Stage 2, pop updated in internal iteration ' num2str(it)]);
    end
    
end