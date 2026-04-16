function img_norm = image_norm_resize(image_path,order)
    % 1. Read the image
    img = imread(image_path); % or .jpg, etc.
    
    % 2. Convert to grayscale (2D)
    if size(img,3) == 3
        img_gray = rgb2gray(img);  % converts RGB to 2D
    else
        img_gray = img;  % already grayscale
    end
    
    % 3. Resize to 700x700
    img_resized = imresize(img_gray, [order order]);
    
    % 4. Normalize to [0,1]
    img_norm = double(img_resized) / 255;  % convert to double first
end