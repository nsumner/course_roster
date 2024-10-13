#!/usr/bin/env python3

import argparse
import glob
import os
import shutil
import subprocess
import sys
import unicodedata
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Optional, TypeAlias, TypeVar

import pandas as pd  # type: ignore[import-untyped]

from . import roster

Grouper: TypeAlias = Callable[[roster.Roster, str, int], None]


#############################################################################
# General Presentation Helpers
#############################################################################

def _strip_diacritics(string: str) -> str:
    normalized = unicodedata.normalize('NFKD', string)
    return "".join(c for c in normalized if not unicodedata.combining(c))


T = TypeVar('T')
def _sorted_by_group(groups: Sequence[tuple[str, T]]) -> Generator[tuple[str,T], None, None]:
    wrapped = [(_strip_diacritics(name), (name, group))
               for name, group in groups]
    wrapped.sort()
    return (grouped for stripped, grouped in wrapped)


#############################################################################
# Actions
#############################################################################

class Action(StrEnum):
    SAVE                    = 'save'
    ASSIGN_GROUPS           = 'assign-groups'
    ASSIGN_ACROSS_GROUPS    = 'assign-across-groups'
    MERGE_FROM_ASSIGNED     = 'merge-from-assigned'
    SHOW_JSON_GROUPS        = 'show-json-groups'
    SHOW_READABLE_GROUPS    = 'show-readable-groups'
    SHOW_EMAILS_FOR_GROUPS  = 'show-emails-for-groups'
    SHOW_CROWDMARK          = 'show-crowdmark'


def _assign_groups(students: roster.Roster,
                   grouper: Grouper,
                   group_column: str,
                   group_size: int) -> None:
    grouper(students, group_column, group_size)


def _assign_across_groups(students: roster.Roster,
                          group_column: str,
                          output_path: str) -> None:
    matching = roster.assign_across_groups(students, group_column)
    matching.to_csv(output_path)

    merged = students.table.merge(matching.table, on=roster.Roster.Field.ID.value)

    groups = _sorted_by_group(merged.groupby(matching.assigned_label))
    for group, members in groups:
        emails = members[roster.Roster.Field.EMAIL.value].tolist()
        print(f'{group}: {", ".join(emails)}')


def _show_json_groups(students: roster.Roster,
                      group_column: str) -> None:
    stubs = roster.get_group_stubs(students, group_column)
    for group, group_line in _sorted_by_group(stubs):
        print(group)
        print(group_line)
        print()


def _show_readable_groups(students: roster.Roster,
                      group_column: str) -> None:
    groups = _sorted_by_group(students.group_by(group_column))
    for group, members in groups:
        print(group, '\n#########')
        names = '\n'.join((members[roster.Roster.Field.PREFERRED_NAME.value]
                           + ' '
                           + members[roster.Roster.Field.LAST_NAME.value]).tolist())
        print(names)
        print()


def _show_emails_for_groups(students: roster.Roster,
                      group_column: str) -> None:
    groups = _sorted_by_group(students.group_by(group_column))
    for count, (group, members) in enumerate(groups):
        ids = ','.join(members[roster.Roster.Field.EMAIL.value].to_list())
        print(f'Group {count} - {group}: {ids}')


def _merge_from_assigned(students: roster.Roster,
                         group_column: str,
                         assigned: roster.GroupMatching,
                         output_dir: str,
                         matching: str) -> None:
    def find_items(username: str) -> list[str]:
        return list(glob.glob(os.path.join(output_dir, username, matching)))

    def collect(students: pd.DataFrame) -> list[str]:
        file_lists = students[roster.Roster.Field.EMAIL.value].map(find_items)
        # Note, because DataFrame.explode turns empty lists into NaNs,
        # we need to filter those out before flattening the column
        non_empty = file_lists[file_lists.apply(len) > 0]
        return non_empty.explode().tolist()  # type: ignore[no-any-return]

    assignments = assigned.table.groupby(assigned.assigned_label)
    collected = [(group, collect(students.table.loc[assigned.index]))
                 for group, assigned in assignments]

    base_pattern, file_type = os.path.splitext(matching)
    match file_type.lower():
        case '.pdf':
            combiner = _combine_pdfs
        case _:
            combiner = _combine_text

    for group, items in collected:
        result_path = os.path.join(output_dir, f'{group}{file_type}')
        combiner(items, result_path)


# File merging helpers. Some of these may rely on outside utilities. If those
# utilities are unavailale, they should throw a RunTimeError.

def _combine_pdfs(sources: list[str], output_path: str) -> None:
    if not sources:
        return

    pdftk_path = shutil.which('pdftk')
    if not pdftk_path:
        raise RuntimeError('Could not find `pdftk` to execute it. PDF merging not available')

    command = [pdftk_path, *sources, 'cat', 'output', output_path]
    subprocess.run(command, shell=False, check=True)  # noqa: S603


def _combine_text(sources: list[str], output_path: str) -> None:
    with open(output_path, 'w') as outfile:
        for source in sources:
            with open(source) as infile:
                outfile.writelines(infile.readlines())


def _show_crowdmark(students: roster.Roster) -> None:
    columns = [
        roster.Roster.Field.LAST_NAME.value,
        roster.Roster.Field.PREFERRED_NAME.value,
        roster.Roster.Field.EMAIL.value,
    ]
    column_rename = {
        roster.Roster.Field.LAST_NAME.value: 'Last Name',
        roster.Roster.Field.PREFERRED_NAME.value: 'First Name',
        roster.Roster.Field.EMAIL.value: 'Email',
    }

    selected = students.table[columns]
    selected.loc[:,roster.Roster.Field.EMAIL.value] = selected[roster.Roster.Field.EMAIL.value] + '@sfu.ca'
    crowdmark = selected.rename(columns=column_rename)
    crowdmark.index.names = ['Student ID Number']
    print(crowdmark.to_csv(index=True))


#############################################################################
# Group creation and naming
#############################################################################

def _create_named_random_grouper(possible_path: str) -> Grouper:
    with open(possible_path) as infile:
        names = [name for line in infile.readlines()
                 if (name := line.strip())]

    def create_named_groups(students: roster.Roster,
                            group_column: str,
                            group_size: int) -> None:
        return roster.group_students_randomly(students,
                                              group_size,
                                              group_column,
                                              names)
    return create_named_groups


def _create_random_groups(students: roster.Roster,
                          group_column: str,
                          group_size: int) -> None:
    roster.group_students_randomly(students, group_size, group_column)


def _parse_coursys_groups(possible_path: str) -> dict[str, str]:
    with open(possible_path) as infile:
        lines = infile.readlines()

    def parse_group_line(line: str) -> tuple[str, list[str]]:
        group_chunk, student_chunk = line.split(': ')

        assert group_chunk.startswith('g-')
        group_name = group_chunk[2:]

        return group_name, student_chunk.strip().split(',')

    group_map = {}
    for line in lines:
        group, students = parse_group_line(line)
        for student in students:
            group_map[student] = group

    return group_map


def _create_coursys_grouper(possible_path: str) -> Grouper:
    coursys_groups = _parse_coursys_groups(possible_path)

    def mapper(uid: str, email: str) -> Optional[str]:
        if email in coursys_groups:
            return coursys_groups[email]
        # If no group is found for the student, we don't want to provide
        # any mapping at all, so None should indicate that in Pandas
        return None

    def create_named_groups(students: roster.Roster,
                            group_column: str,
                            group_size: int) -> None:
        return roster.group_students(students, group_column, mapper)
    return create_named_groups


#############################################################################
# Roster sources
#############################################################################

@dataclass
class RosterInfo:
    students: roster.Roster
    modifiable_source: Optional[str]


def _read_roster_csv(path: str) -> RosterInfo:
    return RosterInfo(roster.from_roster_csv(path), path)


def _read_sims_csv(path: str) -> RosterInfo:
    return RosterInfo(roster.from_sims_csv(path), None)


def _random_roster(size_as_str: str) -> RosterInfo:
    return RosterInfo(roster.from_nothing(int(size_as_str)), None)


#############################################################################
# Main entry point and helpers
#############################################################################


def _get_roster_output(explicit: Optional[str], roster_info: RosterInfo) -> str:
    if explicit:
        return explicit
    if roster_info.modifiable_source:
        return roster_info.modifiable_source
    return 'roster.csv'


def _get_matching_output(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    return 'assigned.csv'


def _get_output_dir(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    return './'


# Any workflow on a roster must have:
#  * a source of roster data
#  * an action to perform on that data
#  * an output target for the results
#
# The source and the output may be implicitly determined, but they are still
# required. If the underlying roster was modified, then it will be saved
# to a roster file after the given operation.

def main() -> None:
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('action', choices=[a.value for a in Action])

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument('--roster',
                              help='',
                              dest='roster',
                              type=_read_roster_csv)
    source_group.add_argument('--sims',
                              help='',
                              dest='roster',
                              type=_read_sims_csv)
    source_group.add_argument('--random-size',
                              help='',
                              dest='roster',
                              type=_random_roster)

    teams_group = parser.add_mutually_exclusive_group(required=False)
    teams_group.add_argument('--group-name-list',
                             help='',
                             dest='grouper',
                             type=_create_named_random_grouper)
    teams_group.add_argument('--coursys-groups',
                             help='',
                             dest='grouper',
                             type=_create_coursys_grouper)

    parser.add_argument('--group-column',
                        help='',
                        type=str,
                        default='Team')

    parser.add_argument('--group-size',
                        help='',
                        type=int,
                        default=8)

    parser.add_argument('--output',
                        help='',
                        type=str)

    parser.add_argument('--assigned',
                        help='',
                        type=str,
                        default='assigned.csv')

    parser.add_argument('--matching',
                        help='',
                        type=str,
                        default='*.pdf')

    try:
        args = parser.parse_args()
    except FileNotFoundError as file_not_found:
        print(f'ERROR: Unable to open and read {file_not_found.filename}.\n',
              file=sys.stderr)
        return

    roster_info = args.roster

    grouper = args.grouper if args.grouper else _create_random_groups
    group_column = args.group_column
    group_size = args.group_size

    output = args.output

    match args.action:
        case Action.SAVE.value:
            output = _get_roster_output(output, roster_info)
            roster.to_roster_csv(roster_info.students, output)

        case Action.ASSIGN_GROUPS.value:
            _assign_groups(roster_info.students, grouper, group_column, group_size)
            output = _get_roster_output(output, roster_info)
            roster.to_roster_csv(roster_info.students, output)

        case Action.ASSIGN_ACROSS_GROUPS.value:
            output = _get_matching_output(output)
            _assign_across_groups(roster_info.students, group_column, output)

        case Action.MERGE_FROM_ASSIGNED.value:
            output = _get_output_dir(output)
            assigned = roster.GroupMatching.from_csv(args.assigned)
            matching = args.matching
            _merge_from_assigned(roster_info.students, group_column, assigned, output, matching)

        case Action.SHOW_JSON_GROUPS.value:
            _show_json_groups(roster_info.students, group_column)

        case Action.SHOW_READABLE_GROUPS.value:
            _show_readable_groups(roster_info.students, group_column)

        case Action.SHOW_EMAILS_FOR_GROUPS.value:
            _show_emails_for_groups(roster_info.students, group_column)

        case Action.SHOW_CROWDMARK.value:
            _show_crowdmark(roster_info.students)


if __name__ == '__main__':
    main()
