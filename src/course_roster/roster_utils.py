#!/usr/bin/env python3

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional, TypeAlias

from . import roster


@dataclass
class Action:
    do: Callable[..., None]
    save_roster: bool


Grouper: TypeAlias = Callable[[roster.Roster, str, int], None]


def _do_nothing(students: roster.Roster,
                grouper: Grouper,
                group_column: str,
                group_size: int) -> None:
    pass


def _assign_groups(students: roster.Roster,
                   grouper: Grouper,
                   group_column: str,
                   group_size: int) -> None:
    grouper(students, group_column, group_size)


def _assign_across_groups(students: roster.Roster,
                          grouper: Grouper,
                          group_column: str,
                          group_size: int) -> None:
    matching = roster.assign_across_groups(students, group_column)
    print(matching.to_csv())


def _show_json_groups(students: roster.Roster,
                      grouper: Grouper,
                      group_column: str,
                      group_size: int) -> None:
    for _, group_line in roster.get_group_stubs(students, group_column):
        print(group_line)


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

def _read_roster_csv(possible_path: str) -> roster.Roster:
    return roster.from_roster_csv(possible_path)


def _read_sims_csv(possible_path: str) -> roster.Roster:
    return roster.from_sims_csv(possible_path)


def _random_roster(size_as_str: str) -> roster.Roster:
    return roster.from_nothing(int(size_as_str))


#############################################################################
# Main entry point
#############################################################################

# Any workflow on a roster must have:
#  * a source of roster data
#  * an action to perform on that data
#  * an output target for the results
#
# The source and the output may be implicitly determined, but they are still
# required. If the underlying roster was modified, then it will be saved
# to a roster file after the given operation. If a different target

def main() -> None:
    core_actions = {
        'save':                  Action(_do_nothing, True),
        'assign-groups':         Action(_assign_groups, True),
        'assign-across-groups':  Action(_assign_across_groups, True),
        'show-json-groups': Action(_show_json_groups, True),
    }
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('action', choices=core_actions.keys())

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

    args = parser.parse_args()

    students = args.roster

    grouper = args.grouper if args.grouper else _create_random_groups

    group_column = args.group_column
    group_size = args.group_size

    action = core_actions[args.action]
    action.do(students, grouper, group_column, group_size)

    if action.save_roster:
        roster.to_roster_csv(students, 'roster.csv')


if __name__ == '__main__':
    main()
