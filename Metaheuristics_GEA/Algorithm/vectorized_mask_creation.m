function [Mask] = vectorized_mask_creation(pop,NFixedX)
    matrix = vertcat(pop.Position1); %% ?
    [rows, cols] = size(matrix);
    Mask = zeros(rows, cols);

    for row_index = 1:rows
        row = matrix(row_index, :);

        diff = bsxfun(@minus, row, matrix);
        zero_mask = (diff == 0);

        zero_count = sum(zero_mask, 1);
        valid_cols = (zero_count >= NFixedX);
        Mask(row_index, valid_cols) = 1;
    end
end

