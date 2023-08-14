# course_roster

Tools for managing team projects in courses. This package provides
libraries and scripts that help with:

* Easy management of students in a course
* Assignment into teams
* Assigning and managing collaboration across teams
* Peer evaluation within teams
* Whole class, team, and individual analytics
* ...

## Example use case

Peer evaluations can be parsed, grouped, and analyzed to produce
quantitative and qualitative analyses of large classes with just:

    analyze-peer-evaluations --evaluation-dir i1evals/
    
synthetic student data can be created and analyzed in the current
directory with:

    analyze-peer-evaluations --generate

## Installing

... is best done within a virtual environment. Either use an existing
environment of your own or create a new one with:

    python3 -m venv --system-site-packages coursesvenv

inside a directory of your choosing. Enter the environment with:

    source coursesvenv/bin/activate

install `course_roster` and its dependencies with:

    cd course_roster
    python -m pip install --editable .[summaries]
    python -c 'import nltk; nltk.download("punkt")'

