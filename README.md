# course_roster

Tools for managing team projects in courses. This package provides
libraries and scripts that help with:

* Easy management of students in a course
* Assignment into teams
* Assigning and managing collaboration across teams
* Peer evaluation within teams
* Whole class, team, and individual analytics
* ...

## Example use cases

### Creating teams from a [SIMS](https://go.sfu.ca) roster CSV

... can be done with:

    roster-utils assign-groups --sims SFU_SRCLSRST.CSV --group-name-list group-names.txt
    
where `SFU_SRCLSRST.CSV` is the SIMS exported roster file. An instructor can
get the SIMS roster via the "Class Roster / Email" button at the bottom of
the SIMS enrollement page for a course section. `group-names.txt` is a text
file with each line containing an assigned name for a group. This will produce
a `Roster` formatted CSV called `roster.csv` containing the core information
about the students along with a column `Team` containing the name of the group
to which each student was assigned. e.g.

| Student ID | Last Name | Preferred Name | Email   | Plan    | Team     |
|------------|-----------|----------------|---------|---------|----------|
| 900000000  | Doe       | John           | doej    | CMPTMAJ | Baklava  |
| 900000001  | Smith     | Dan            | smithd  | SOSYMAJ | Cake     |
| 900000002  | Garcia    | Jerry          | garciaj | CMPTMAJ | Cake     |
| 900000003  | Nguyen    | Duy            | nguyend | SOSYMAJ | Affogato |
| ...        | ...       | ...            | ...     | ...     | ...      |

Assigning groups based on CourSys exported group information can be done with

    roster-utils assign-groups --sims SFU_SRCLSRST.CSV --coursys-groups coursys-groups.txt

An existing `roster.csv` can also be used, and the group name argument can be
omitted if you do not care.

    roster-utils assign-groups --roster roster.csv

By default the number of students per group is 8, and the number of groups will
split the students roughly evenly with that as the maximum size. The group size
and column name can also be configured:

    roster-utils assign-groups --roster roster.csv --group-column "Project 1" --group-size 4
    
This allows assigning students to multiple groups over a semester and
maintaining information across all of them.


### Peer evaluations

... can be managed, parsed, and analyzed. An example peer evaluation form is
provided in [shared/html/peer-evaluation.html](shared/html/peer-evaluation.html).
The contents of the form are customized per group via a JSON stub that the
student can enter.

To produce JSON stubs for the groups in your course, you can run

    roster-utils show-json-groups --roster roster.csv
    
A student can then complete the peer evaluation, which produces a JSON fragment
for them to download and submit via CourSys, Canvas, or any othe medium.
Submitted peer evaluations can then be parsed, grouped, and analyzed to produce
quantitative and qualitative analyses of large classes with just:

    analyze-peer-evaluations --evaluation-dir i1evals/

where `--evaluation-dir` identifies a directory containing submissions, with
each student's evaluation in a directory named after the student's SFU
username. Synthetic student data can be created and analyzed in the current
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

