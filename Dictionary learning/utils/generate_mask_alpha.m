function [mask, r_factor] = generate_mask_alpha(siz, r_factor_designed, r_alpha, axis_undersample,acs, seed, mute)
    % init
    mask = zeros(siz);
    if seed >= 0
        rng(seed)
    end
    % get samples
    num_phase_encode = siz(axis_undersample);
    num_phase_sampled = int16(floor(num_phase_encode / r_factor_designed));
    % coordinate
    coordinate_normalized = 1:num_phase_encode;
    coordinate_normalized = abs(coordinate_normalized - num_phase_encode / 2) / (num_phase_encode / 2.0);
    prob_sample = coordinate_normalized .^r_alpha;
    prob_sample = prob_sample / sum(prob_sample);
    % sample
    index_sample = datasample(1:num_phase_encode, num_phase_sampled,...
                                    'Replace',false, 'Weights',prob_sample);
    % sample
    if axis_undersample == 0
        mask(index_sample, :) = 1;
    else
        mask(:, index_sample) = 1;
    end
%     mask_temp = np.zeros_like(mask);
    
    % acs
    if axis_undersample == 1
        mask(floor(1:(acs + 1) / 2), :) = 1;
        mask(floor(end-acs / 2):end, :) = 1;
    else
        mask(:, floor(1:(acs + 1) / 2)) = 1;
        mask(:, floor(end-acs / 2):end) = 1;
    end
    mask = fftshift(mask);
    % compute reduction
    r_factor = length(mask(:)) / sum(mask(:));
    if ~mute
        fprintf('gen mask size of (%d,%d) for R-factor= %f\n', size(mask),r_factor)
%         print(num_phase_encode, num_phase_sampled, np.where(mask[0, :]))
    end
