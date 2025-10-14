function m = sigma_delta_RBBMM(x,ref,ECA2)
        
        % x is the image
        % ref is the reference signal for comparator
        % ECA2 is pseudo random sequences of length 256 each.
        % ECA sequence is extracted from a longer sequence of 512 cells
        % of Rule30 ECA.

        % ECA2 is utilized for random modulation equivalent to beta bits

       nc = size(x,2);    % number of columns in image
        nr = size(x,1);    % number of rows in image
        col = zeros(nc,1); % vector initialized to hold the CS sample generated in each column
        f0=255;            % maximum feedback value applied when the comparator is set 

        aref = ref;      % comparator reference voltage
        for j=1:nc       % loop on number of columns of image
            x1 = x(:,j); % extract the jth column of the image
           
            a0 = 0;      % initialize the accumulator output
            f = 0;       % initialize the feedback

            Dig_out = zeros(nr,1); % this holds the counter's output as the rows are readout
            k=0; % intermediate variable to hold the counter's output
            
            for i=1:nr % This is a loop across the rows for the selected column in the first loop
               
                    a = x1(i) - f;  % from the input pixel value substract the feedback
                    f=0;            % set the feedback to zero
                    a1 = a+a0;      % accumulator's present output=new_input(a)+past value (a0)
                    a0 = a1;        % store the present value of accumulator
                    
                    c = a1-aref; % output of the comparator=accumulator's output-reference

                    if c>=0      % if comparator is set 
                        f = f0;  % set the feedback to maximum value

                        if ECA2(i,j)==1   % if the beta=1
                            Dig_out(i) = k +1; % count up
                            k = Dig_out(i);  % store the intermediate count value
                        else
                            Dig_out(i) = k -1;  % if beta=0 then count down
                            k = Dig_out(i);    % store the intermediate count value
                        
                        end
                    else
                        Dig_out(i) = k;  % keep the count same as previous
                       
                   
                    end
            end

            col(j) = k;  % final count value assigned to corresponding row vector
        end
        %m = sum(col);
        m = col;  % vector having final count values of all columns is assigned to feature vector
end