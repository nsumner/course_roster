#!/usr/bin/env python3

import os
import random
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from json import dumps as dump_json
from math import ceil
from typing import Optional, Union

import pandas as pd  # type: ignore[import]

#############################################################################
# Core Roster
#############################################################################

IDList = list[str]
GroupName = str
Path = Union[str, bytes, os.PathLike]


@dataclass
class Roster:
    class Field(Enum):
        ID             = 'Student ID'  # only used as an index
        LAST_NAME      = 'Last Name'
        PREFERRED_NAME = 'Preferred Name'
        EMAIL          = 'Email'
        PLAN           = 'Plan'

    table: pd.DataFrame

    def get_email_ids(self) -> list[str]:
        return self.table[Roster.Field.EMAIL.value].tolist()  # type: ignore[no-any-return]

    def get_ids(self) -> list[str]:
        return self.table.index.values.tolist()  # type: ignore[no-any-return]

    def group_by(self, column: str) -> pd.core.groupby.DataFrameGroupBy:
        return self.table.groupby(column)


def _make_last_name(full_name: str) -> str:
    return full_name[:full_name.index(',')]


def _email_to_username(address: str) -> str:
    if address.endswith('@sfu.ca'):
        return address[:-7]
    if address.endswith('@alumni.sfu.ca'):
        return address[:-14]
    return address


def from_sims_csv(sims_csv_path: Path) -> Roster:
    sims_frame = pd.read_csv(sims_csv_path)

    # Most of the SIMS roster is irrelevant, so select only a few
    # parts to retain. If another field/column becomes relevant,
    # it should be added to the list
    relevant_labels = [
        'student_ID',         # numerical student ID
        'student_name',       # full student name in form "Last, First"
        'student_PRF_Fname',  # preferred name
        'Campus_Email',       # SFU email address
        'acad_plan',          # Academic plan, e.g. "SOSYMAJ/SYSONE"
    ]
    columns_to_keep = sims_frame.columns.difference(relevant_labels)
    sims_frame.drop(columns=columns_to_keep, inplace=True)

    sims_frame['student_name'] = sims_frame['student_name'].map(_make_last_name)
    sims_frame['Campus_Email'] = sims_frame['Campus_Email'].map(_email_to_username)

    new_labels = {
        'student_ID':        Roster.Field.ID.value,
        'student_name':      Roster.Field.LAST_NAME.value,
        'student_PRF_Fname': Roster.Field.PREFERRED_NAME.value,
        'Campus_Email':      Roster.Field.EMAIL.value,
        'acad_plan':         Roster.Field.PLAN.value,
    }
    sims_frame = sims_frame.rename(columns=new_labels)
    sims_frame.set_index(Roster.Field.ID.value, inplace=True, drop=True)

    return Roster(sims_frame)


def from_roster_csv(roster_csv_path: Path) -> Roster:
    roster_dataframe = pd.read_csv(roster_csv_path)
    assert {field.value for field in Roster.Field}.issubset(roster_dataframe.columns)

    roster_dataframe.set_index(Roster.Field.ID.value, inplace=True, drop=True)

    return Roster(roster_dataframe)


# This generates a fake Roster of a given size. This is useful, for instance
# when testing out different tools or workflows that you may want to use in
# a course.
def from_nothing(size: int) -> Roster:
    plans = ['SOSYMAJ', 'CMPTMAJ']
    fake_dataframe = pd.DataFrame({
        Roster.Field.ID.value:             [str(900000000 + i) for i in range(size)],
        Roster.Field.LAST_NAME.value:      ['L' + str(i) for i in range(size)],
        Roster.Field.PREFERRED_NAME.value: ['P' + str(i) for i in range(size)],
        Roster.Field.EMAIL.value:          ['user' + str(i) for i in range(size)],
        Roster.Field.PLAN.value:           [random.choices(plans, k=1)[0] for i in range(size)],
    })
    fake_dataframe.set_index(Roster.Field.ID.value, inplace=True, drop=True)
    return Roster(fake_dataframe)


def to_roster_csv(roster: Roster, path: Path) -> None:
    roster.table.to_csv(path, index=True)


#############################################################################
# Group Management and Collaborations
#############################################################################

def group_students(roster: Roster,
                   label: str,
                   namer: Callable[[str, str], str]) -> None:
    def wrapper(x: pd.Series) -> str:
        return namer(x.name, x.Email)

    roster.table[label] = roster.table.apply(wrapper, axis=1)


def group_students_randomly(
        roster: Roster,
        group_size: int,
        label: str,
        group_labels: Optional[list[GroupName]] = None) -> None:
    students = roster.get_ids()

    random.shuffle(students)
    num_groups = int(ceil(len(students) / group_size))
    group_list = [students[group:len(students):num_groups]
                  for group in range(num_groups)]

    def name_mapping(group_id: int) -> str:
        if group_labels and group_id < len(group_labels):
            return group_labels[group_id]
        return str(group_id)

    group_map = {student: name_mapping(group_id)
                 for group_id, group in enumerate(group_list)
                 for student in group}

    group_students(roster, label, lambda id, username: group_map[id])


@dataclass
class GroupMatching:
    # contains columns for:
    #   *index*: Roster.Field.ID.value: The id of the student
    #   group_label: The label of the main group of the student
    #   assigned_label: The label of the group the student will collaborate with
    table: pd.DataFrame
    group_label: str
    assigned_label: str


# Given a roster and column label for groups, assign each student another
# group to collaborate with. Students should be assigned roughly uniformly
# across the groups.
def assign_across_groups(roster: Roster, group_label: str) -> GroupMatching:
    groups = set(roster.table[group_label].unique())
    assert len(groups) > 1
    target_count = len(roster.table.index) // len(groups)

    # The result will be a new frame with just
    # [*index*: Student ID, Original Group, Assigned Group]
    df = roster.table[[group_label]]
    df = df.assign(Assigned=lambda x: '')
    assigned_label = 'Assigned'

    # TODO: Do we want uniform permutations so that groups are consistently
    # "cross pollinated"? The elow would work fine after hitting a maximum
    # number of permutations based on the roster.

    # First, sample enough to ensure all groups get collaborators
    for group_name in groups:
        # For mutability, we have to construct a sample over indices with pandas.
        # The sample comes from all students not in the given group who have not
        # yet been assigned another group to collaborate with.
        sources = df.index[(df[group_label] != group_name) & (df[assigned_label] == '')]
        sample_size = min(target_count, len(sources))
        sample_indices = random.sample(sources.tolist(), sample_size)
        df.loc[sample_indices, assigned_label] = group_name

    # The clean up by making sure that all students are assigned collaboration.
    # As a semester progresses, this can get messier as students drop and group
    # sizes become uneven. For now, simply assign each remaining student.
    # Prefer not to reuse a group if possible.
    leftovers = df.index[df[assigned_label] == '']
    used_groups: set[str] = set()
    for student in leftovers:
        original_group = df.loc[student][group_label]
        possible_groups = groups.difference({original_group}).difference(used_groups)
        if not possible_groups:
            possible_groups = groups.difference({original_group})
        (assigned_group,) = random.sample(list(possible_groups), 1)
        df.loc[student, assigned_label] = assigned_group
        used_groups.add(assigned_group)

    assert df.loc[df[assigned_label] == ''].empty

    return GroupMatching(df, group_label, assigned_label)


def from_matching_assignment_csv(roster_csv_path: Path) -> GroupMatching:
    matching_dataframe = pd.read_csv(roster_csv_path)
    columns = list(matching_dataframe.columns)
    expected_column_count = 3
    assert len(columns) == expected_column_count
    assert Roster.Field.ID.value in columns

    matching_dataframe.set_index(Roster.Field.ID.value, inplace=True, drop=True)

    return GroupMatching(matching_dataframe, columns[1], columns[2])


def to_matching_assignment_csv(matching: GroupMatching, path: Path) -> None:
    matching.table.to_csv(path, index=True)


def get_group_stubs(roster: Roster,
                    group_label: str) -> list[tuple[GroupName, str]]:
    groups = roster.group_by(group_label)

    def get_student_stub(student: pd.Series) -> dict[str, str]:
        last_name = student[Roster.Field.LAST_NAME.value]
        preferred_name = student[Roster.Field.PREFERRED_NAME.value]
        return {
            'name': last_name + ', ' + preferred_name,
            'email': student[Roster.Field.EMAIL.value],
        }

    def get_stub(group: pd.Series) -> str:
        return dump_json(group.apply(get_student_stub, axis=1).to_list())

    stubs = [(group_id, get_stub(group)) for group_id, group in groups]
    stubs.sort()
    return stubs
