function projRoot = helperCSINetExtractProject(projectName)

projRoot = ...
  matlab.internal.project.archive.extractArchive( ...
  strcat(projectName,".mlproj"), ...
  'ExtractionFolder',pwd, ...
  'OpenAfterExtraction',false);
