function Pitch = Complex_Frequency_based_Pitch_estimation(Speech, VAD)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Yuya HOSODA, Arata KAWAMURA, Youji IIGUNI,                                                   
% "Complex Frequency-Domain Pitch Estimation for Narrowband Speech Signals,"  
% IEEE/ACM Trans. Audio, Speech, Lang. Process., 2021. (to be submitted)           
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Input
%  |-Speech...Speech signal at a sampling rate of 8 kHz
%  |-VAD...Voiced active frame
%
% Output
%  |-Pitch...Estimated Pitch
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%          
% Variable 
%          
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Sampling rate
Fs = 8000;

% Frame length (default 40ms)
FrameLength = 0.040*Fs;

% Hop size (default 10ms)
HopSize = 0.010*Fs;

% Parameters for Transition propability
Parameter_TransitionPropability = [120, 0.3679];

% Pole for LPC
Pole_LPC = 12;

% Number of the complex spectra of harmonics to be averaged
Number_HarmonicAveraged = 3;

% Number of the cumulative sum to be temporal smoothed
Number_FrameAveraged = 3;

% Number of analysis frames
Number_Frame = length(VAD);

% Pitch range of 50-400 Hz
Pitch_Min = 50;
Pitch_Max = 400;
Pitch_Candidate = Pitch_Min:Pitch_Max;

% Harmonic Number
Harmonic_Number = 0:Number_HarmonicAveraged-1;

% Subharmonic Number
Subharmonic_Number_Twice = 1/2:1:Number_HarmonicAveraged-1;
Subharmonic_Number_Third = [1/3:1:(Number_HarmonicAveraged-1)/2, 2/3:1:(Number_HarmonicAveraged-1)/2];

% Harmonics on the narrowband (>300Hz)
Pitch_Base = ceil(300./Pitch_Candidate).*Pitch_Candidate;
Harmonic_Hz = Harmonic_Number'*Pitch_Candidate + Pitch_Base;

% Subharmonic on the narrowband (>300Hz)
Subharmonic_Twice_Hz = Subharmonic_Number_Twice'*Pitch_Candidate + Pitch_Base;
Subharmonic_Third_Hz = Subharmonic_Number_Third'*Pitch_Candidate + Pitch_Base;

% Frequency index of harmonics on the narrowband
Harmonic_Index = round(Harmonic_Hz./(Fs./FrameLength))+1;
Subharmonic_Twice_Index = round(Subharmonic_Twice_Hz./(Fs./FrameLength))+1;
Subharmonic_Third_Index = round(Subharmonic_Third_Hz./(Fs./FrameLength))+1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%          
% Initialization
%          
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Generate margin
Speech = [Speech; zeros(FrameLength, 1)];

% Flag for the Viterbi algorithm
Vitabi_Frag = -1;

% Pitch candidate in the past frame
Pitch_Candidate_Past = Pitch_Candidate;

% Transition Propability
TransitionProbability_Base = @(x, xdata)max(1-(abs(xdata/x(1))).^(x(2)), eps);
PitchDifferrence_CentScall = abs(Pitch_Candidate'-Pitch_Candidate_Past);
TransitionProbability_Function = TransitionProbability_Base(Parameter_TransitionPropability, PitchDifferrence_CentScall);

% Matrix for cumulative sum
CumulativeSum_All = zeros(Number_Frame, length(Pitch_Candidate));

% Estimated pitch
Pitch = zeros(Number_Frame, 1);

%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Proposed algorithm   %
%%%%%%%%%%%%%%%%%%%%%%%%%%

for Frame_Index = 1:Number_Frame
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %   Pitch candidate selection using YIN algorithm   %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    
    % Input to be analyzed
    Input_Analyzed = Speech((Frame_Index-1)*HopSize+1:(Frame_Index-1)*HopSize+FrameLength);  

    % Analisys window function
    if rem(FrameLength, 2)==0
        WindowFunction = hann(FrameLength, 'periodic');
    else
        WindowFunction = hann(FrameLength);
    end
    
    % DFT
    DFT = fft(WindowFunction.*Input_Analyzed);
    
    % Estimate pitch when the analyzed frame is active
    if VAD(Frame_Index)>0   
        
        % Avoid LPC calculation error
        if sum(abs(Input_Analyzed))<eps 
            Input_Analyzed = Input_Analyzed + eps*randn(FrameLength, 1);
        end

        % Calculate LPC
        AR_Coefficient = lpc(Input_Analyzed, Pole_LPC);

        % Calculate an excitation signal
        Excitation = Input_Analyzed - filter([0 -AR_Coefficient(2:end)], 1, Input_Analyzed);

        % Amplitude spectrum of the excitation signal
        Magnitude_Excitation = abs(fft(Excitation.*WindowFunction));
    
        % Amplitude spectrum of harmonics and subharmonics
        Magnitude_Excitation_Harmonic = Magnitude_Excitation(Harmonic_Index);
        Magnitude_Excitation_Subharmonic_Twice = Magnitude_Excitation(Subharmonic_Twice_Index);
        Magnitude_Excitation_Subharmonic_Third = Magnitude_Excitation(Subharmonic_Third_Index);
    
        
        %%%%%%%%%%%%%%%%%%%%%%%%
        %   Phase difference   %
        %%%%%%%%%%%%%%%%%%%%%%%%

        % Calculate Phase difference after 2nd frame
        if Frame_Index>1
            % DFT for harmonics and subharmonics 
            DFT_Harmonic = DFT(Harmonic_Index);
            DFT_Subharmonic_Twice = DFT(Subharmonic_Twice_Index);
            DFT_Subharmonic_Third = DFT(Subharmonic_Third_Index);
        
            % Phase spectrum of harmonics and subharmonics at a past frame
            Phase_Harmonic_Past = angle(DFT_Past(Harmonic_Index));
            Phase_Subharmonic_Twice_Past = angle(DFT_Past(Subharmonic_Twice_Index));
            Phase_Subharmonic_Third_Past = angle(DFT_Past(Subharmonic_Third_Index));

            % Phase difference between harmonic and subharmonics
            PhaseDifference_Harmonic = Phase_Harmonic_Past(2:end, :) - Phase_Harmonic_Past(1, :) + 2*pi*Harmonic_Number(2:end)'*Pitch_Candidate*HopSize/Fs;
            PhaseDifference_SubHarmonic_Twice = Phase_Subharmonic_Twice_Past - Phase_Harmonic_Past(1, :) + 2*pi*Subharmonic_Number_Twice'*Pitch_Candidate*HopSize/Fs;
            PhaseDifference_SubHarmonic_Third = Phase_Subharmonic_Third_Past - Phase_Harmonic_Past(1, :) + 2*pi*Subharmonic_Number_Third'*Pitch_Candidate*HopSize/Fs;

            % Modify the phase spectrum of harmonics and subharmonics
            % using phase difference between harmonic and subharmonics
            Phase_Harmonic_PhaseDifference = angle(DFT_Harmonic);
            Phase_Harmonic_PhaseDifference(2:end, :) = Phase_Harmonic_PhaseDifference(2:end, :) - PhaseDifference_Harmonic;
            Phase_Subharmonic_PhaseDifference_Twice = angle(DFT_Subharmonic_Twice);
            Phase_Subharmonic_PhaseDifference_Twice = Phase_Subharmonic_PhaseDifference_Twice - PhaseDifference_SubHarmonic_Twice;
            Phase_Subharmonic_PhaseDifference_Third = angle(DFT_Subharmonic_Third);
            Phase_Subharmonic_PhaseDifference_Third = Phase_Subharmonic_PhaseDifference_Third - PhaseDifference_SubHarmonic_Third;

            % Generate a complex spectrum
            ComplexSpectrum_Harmonic = Magnitude_Excitation_Harmonic .* exp(1i * Phase_Harmonic_PhaseDifference);
            ComplexSpectrum_Subharmonic_Twice = Magnitude_Excitation_Subharmonic_Twice .* exp(1i * Phase_Subharmonic_PhaseDifference_Twice);
            ComplexSpectrum_Subharmonic_Third = Magnitude_Excitation_Subharmonic_Third .* exp(1i * Phase_Subharmonic_PhaseDifference_Third);
        
            % Calculate a cummulative sum
            CumulativeSum = abs(sum(ComplexSpectrum_Harmonic))-abs(sum(ComplexSpectrum_Subharmonic_Twice)) - abs(sum(ComplexSpectrum_Subharmonic_Third));
        
            % Normalization
            CumulativeSum = (CumulativeSum - min(CumulativeSum));
            CumulativeSum = CumulativeSum./max(CumulativeSum);
        
            % Record the normalized cummulative sum
            CumulativeSum_All(Frame_Index, :) = CumulativeSum;

            % Temporal smoothing
            TemporalSmoothed_CumulativeSum = mean(CumulativeSum_All(max(1, Frame_Index-Number_FrameAveraged):Frame_Index, :));

            % The picth candidates exsisted in the previous frame
            if Vitabi_Frag>0

                % Transition Propability
                if length(Pitch_Candidate_Past)==1
                    TransitionProbability = TransitionProbability_Function(:, Pitch_Candidate_Past-Pitch_Min+1);
                else
                    TransitionProbability = TransitionProbability_Function.*Viterbi_Score_Past';
                end

                % Calculate a Viterbi score
                Viterbi_Score = max(TransitionProbability, [], 2).*TemporalSmoothed_CumulativeSum';

                % Determine pitch
                if VAD(Frame_Index)>0
                    [~, Pitch_index] =  max(Viterbi_Score);
                    Pitch(Frame_Index, 1) = Pitch_Candidate(Pitch_index);

                    % Update pitch candidates and Viterbi score
                    Pitch_Candidate_Past = Pitch_Candidate;
                    Viterbi_Score_Past = Viterbi_Score./sum(Viterbi_Score);
                end
            else
                if VAD(Frame_Index)>0         
                    [~, LL] = max(TemporalSmoothed_CumulativeSum);
                    Pitch(Frame_Index, 1) = Pitch_Candidate(LL);

                    % Update pitch candidates and Viterbi score
                    Pitch_Candidate_Past = Pitch(Frame_Index, 1);
                    Viterbi_Score_Past = 1;
                
                    Vitabi_Frag = 1;
                end
            end
        end        
    end

    
    % Phase spectrum for NB harmonics
    DFT_Past = DFT;
    
end
