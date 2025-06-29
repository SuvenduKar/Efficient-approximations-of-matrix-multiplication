function B = block_toeplitz(matrix_size, block_size)
    % Ensure matrix_size is divisible by block_size
    if mod(matrix_size, block_size) ~= 0
        error('matrix_size must be divisible by block_size.');
    end

    % Number of blocks along one dimension
    num_blocks = matrix_size / block_size;

    % Create the first row of blocks with entries from Uniform(0,1)
    block_row = cell(1, num_blocks);
    for k = 1:num_blocks
        block_row{k} = rand(block_size);  % Entries ~ Uniform(0,1)
    end

    % Initialize the full block Toeplitz matrix
    B = zeros(matrix_size);

    % Fill in the blocks using Toeplitz structure
    for i = 1:num_blocks
        for j = 1:num_blocks
            % Block index follows Toeplitz pattern
            if j >= i
                block = block_row{j - i + 1};
            else
                block = block_row{i - j + 1};
            end

            % Insert the block into the matrix
            row_range = (i-1)*block_size + 1 : i*block_size;
            col_range = (j-1)*block_size + 1 : j*block_size;
            B(row_range, col_range) = block;
        end
    end
end
