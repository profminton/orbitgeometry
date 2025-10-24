#!/bin/bash
jupyter nbconvert\
   --to html\
   --output-dir html\
   --TemplateExporter.exclude_input=True\
   --no-prompt\
   $1
