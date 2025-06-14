%% testUtilScope.m
% Verify that helloUtil (in utils/) can be called after adding it to the path.

% 1) Determine projectRoot:
thisFile = mfilename("fullpath");  
if ~isempty(thisFile)
    % If running as a function, mfilename returns the full path 
    [projectRoot, ~, ~] = fileparts(thisFile);
else
    % If running as a script, fallback to current folder:
    projectRoot = pwd;
end

% 2) Look for utils/ in two possible locations:
utilsPath1 = fullfile(projectRoot, "utils");
utilsPath2 = fullfile(projectRoot, "experiments", "utils");

if     isfolder(utilsPath1)
    utilsFolder = utilsPath1;
elseif isfolder(utilsPath2)
    utilsFolder = utilsPath2;
else
    error( ...
      "Could not find a `utils` folder under the project root.\n" + ...
      "Checked:\n" + ...
      "  %s\n" + ...
      "  %s\n" + ...
      "Make sure helloUtil.m lives in one of those locations.", ...
      utilsPath1, utilsPath2);
end

% 3) Add utilsFolder to MATLAB path:
addpath(utilsFolder);
fprintf("Added utils folder to path:\n    %s\n\n", utilsFolder);

% 4) Confirm helloUtil.m is on the path:
if exist("helloUtil", "file") == 2
    disp("helloUtil is now on the path.");
else
    error("helloUtil.m not found on the path after addpath(%s).", utilsFolder);
end

% 5) Call helloUtil to confirm it actually runs:
greeting = helloUtil("Tariq");
disp("Function call succeeded. Output:");
disp(greeting);
