#!/bin/bash
set -euo pipefail

# Runs before every Write/Edit. When the target carries user-facing prose
# (docs, docstrings, comments, README, website pages), inject a reminder that
# the writing style guide is mandatory: it must be read in full this session
# and the text must follow it. Stays silent for other files so code-only or
# config-only edits do not add noise.

path="$(jq -r '.tool_input.file_path // empty')"
[ -z "$path" ] && exit 0

case "$path" in
  *.md|*.qmd|*.markdown|*.rst|*.txt|*.py|*README*|*/docs/*|*/devdocs/*)
    cat <<'JSON'
{"hookSpecificOutput":{"hookEventName":"PreToolUse","additionalContext":"Mandatory: this file carries user-facing prose governed by devdocs/guidance/writing_style.md. You must have read that guide in full this session and make the prose here follow it: the methods-appendix voice for a health economics modeler, lead with the point, no em-dashes or exclamation marks, sentence-case headings, every acronym spelled out on first use, and a revision pass (more than one read) before you consider it done. If you have not read the guide yet this session, read it now before writing."}}
JSON
    ;;
esac
exit 0
