function m = sigma_delta_RBMM(x,ref,ECA1)
        % x is the image
        %ref is the reference signal for comparator
        % ECA1 and ECA2 are pseudo random sequences of length 256 each.
        % These sequences are extracted from a longer sequence of 512 cells
        % of Rule30 ECA.
        % ECA1 is utilized for the pixel selectioon equivalent to alpha
        % bits
        % ECA2 is utilized for random modulation equivalent to beta bits

        nc = size(x,2);    % number of columns in image
        nr = size(x,1);    % number of rows in image
        col = zeros(nc,1); % vector initialized to hold the CS sample generated in each column
        f0=255;            % maximum feedback value applied when the comparator is set 

        aref = ref;      % comparator reference voltage
        for j=1:nc       % loop on number of columns of image
            x1 = x(:,j); % extract the jth column of the image
            sel = find(ECA1(:,j)==0); % Find the indices in jth column
                         % where the ECA1 sequence has zero. 
                         % This will help to skip the corresponding pixels. 
            a0 = 0;      % initialize the accumulator output
            f = 0;       % initialize the feedback

            Dig_out = zeros(nr,1); % this holds the counter's output as the rows are readout
            k=0; % intermediate variable to hold the counter's output
            
            for i=1:nr % This is a loop across the rows for the selected column in the first loop
                if ~ismember(i,sel) % if i is not the index of zero value then execute the code below
                                    % otherwise skip the processing and change i
           
                    a = x1(i) - f;  % from the input pixel value substract the feedback
                    f=0;            % set the feedback to zero
                    a1 = a+a0;      % accumulator's present output=new_input(a)+past value (a0)
                    a0 = a1;        % store the present value of accumulator
                    
                    c = a1-aref; % output of the comparator=accumulator's output-reference

                    if c>=0      % if comparator is set 
                        f = f0;  % set the feedback to maximum value       
                        Dig_out(i) = k +1; % count up
                        k = Dig_out(i);  % store the intermediate count value

                    else
                        f=0; %if comparator is not set keep the feedback to 0
                        Dig_out(i) = k;  % keep the count same as previous
                       
                    end
                end
            end

            col(j) = k;  % final count value assigned to corresponding row vector
        end
        %m = sum(col);
        m = col;  % vector having final count values of all columns is assigned to feature vector
end