#!/bin/bash
jupyter nbconvert\
   --to markdown\
   --output-dir . \
   --TemplateExporter.exclude_input=True\
   $1