#!/usr/bin/env python3

import argparse
import datetime
import heapq
import json
import logging
import os
import re
import string
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import seaborn as sns
from pydriller import Commit, ModifiedFile, Repository  # type: ignore[import-not-found]

logger = logging.getLogger('GitActivity')


@dataclass
class _ModificationInfo:
    lines_added: int
    methods_changed: int
    issues: int

    def __add__(self, other: '_ModificationInfo') -> '_ModificationInfo':
        return _ModificationInfo(self.lines_added + other.lines_added,
                                self.methods_changed + other.methods_changed,
                                self.issues + other.issues)

@dataclass
class _CommitInfo:
    modified: _ModificationInfo
    authors: list[str]
    timestamp: datetime.datetime
    hash: str

@dataclass
class _ProjectConfiguration:
    should_count: Callable[[ModifiedFile], bool]
    passes_quality: Callable[[ModifiedFile], bool]
    count_nontrivial: Callable[[ModifiedFile], int]


def _extract_modification_stats(modification: ModifiedFile,
                                configuration: _ProjectConfiguration) -> _ModificationInfo:
    lines_added     = configuration.count_nontrivial(modification.diff_parsed['added'])
    methods_changed = len(modification.changed_methods)
    passes_quality  = configuration.passes_quality(modification) # noqa: F841
    return _ModificationInfo(lines_added, methods_changed, 0)


def _extract_authors_from_commit(commit: Commit) -> list[str]:
    authors = [commit.author.email]

    for raw_line in commit.msg.splitlines():
        line = raw_line.strip().lower()

        if line.startswith(('author:', 'authors:')):
            authors = [author.strip() for author in line[8:].split(',')]
            logger.info('Found author list: %s', line)

        elif line.startswith('co-authored-by:') and '<' in line:
            # Co author lines have the form:
            #   name <###+username@users.noreply.github.sfu.ca>
            name_start = line.index('<') + 1
            name_end = line.index('>')
            name = line[name_start: name_end]
            authors.append(name)
            logger.info('Found GitHub co-author: %s', name)

        elif '@' in line or 'pair' in line or 'Pair' in line:
            logger.warning('Found unexpected @: %s', line)

    authors = list(set(authors))

    logger.debug('Extracted authors: %.7s, %s, %s',
                 commit.hash, authors, commit.msg.replace('\n', '-'))

    return authors


def _extract_contributions_from_commit(commit: Commit,
                                       configuration: _ProjectConfiguration) -> Optional[_CommitInfo]: # noqa: E501
    modifications = [m for m in commit.modified_files if configuration.should_count(m)]
    if not modifications:
        return None

    timestamp = commit.author_date
    authors = _extract_authors_from_commit(commit)

    mod_info = sum((_extract_modification_stats(modification, configuration)
                   for modification in modifications),
                   start=_ModificationInfo(0,0,0))

    return _CommitInfo(mod_info, #.lines_added, mod_info.methods_changed, mod_info.issues,
                      authors, timestamp, commit.hash)


# This function aggregates the contributions from different commits into a
# single summary of the overall contributions to the repo.
#    Callable[[ModifiedFile], bool]
def _extract_commits_from_repo(repo: Repository,
                               configuration: _ProjectConfiguration) -> list[_CommitInfo]:
    commits = (_extract_contributions_from_commit(commit, configuration)
               for commit in repo.traverse_commits())
    return [commit for commit in commits if commit]


def _compute_per_student(commits: list[_CommitInfo],
                         alias_map: dict[str, str]) -> dict[str, list[_CommitInfo]]:
    student_data = defaultdict(list)
    for commit in commits:
        for student in commit.authors:
            name = alias_map.get(student, student)
            student_data[name].append(commit)
    return student_data


def _contributed_lines(commit: _CommitInfo) -> float:
    return commit.modified.lines_added/float(len(commit.authors))


_FILTERS: dict[Optional[str], Callable[[ModifiedFile], bool]] = {
    'cpp':  lambda m: any(m.filename.lower().endswith(suffix)
                          for suffix in ('.h', '.hpp', '.cpp', '.cxx', '.cc', '.c')),
    'java': lambda m: m.filename.lower().endswith('.java'),
    None:   lambda m: True,
}


# NOTE: A preferred approach would be to apply a semantic diff, remove comments, and observe
#  additions that way
def _count_nontrivial_changes_in_cpp(changes: list[ModifiedFile]) -> int:
    translator = str.maketrans('', '', string.punctuation + string.whitespace)
    def is_nontrivial(line: str) -> bool:
        line = line.strip()
        return line != '' \
          and not line.startswith('//')\
          and not line.startswith('#')\
          and line.translate(translator) != ''
    return sum(1 for (count, line) in changes if is_nontrivial(line))


@dataclass
class _PlottableInfo:
    student: str
    count: float
    bucket: int


_DAYS_PER_TICK = 5

def _plot_contributions(commits: list[_CommitInfo],
                        begin_time: Optional[datetime.datetime],
                        end_time: Optional[datetime.datetime]) -> None:
    if not commits:
        logger.warning('There are no commits to plot!')
        return
    if not begin_time:
        begin_time = commits[0].timestamp
    if not end_time:
        end_time = commits[-1].timestamp

    number_of_days = (end_time - begin_time).days

    if number_of_days < _DAYS_PER_TICK:
        logger.warning('Too few days of active work to plot commits over time')
        return

    as_list = [_PlottableInfo(student,
                              _contributed_lines(commit),
                              (commit.timestamp - begin_time).days)
                 for commit in commits
                 for student in commit.authors]
    as_map = {
        'student': [info.student for info in as_list],
        'count': [info.count for info in as_list],
        'bucket': [info.bucket for info in as_list],
      }

    sns.set_theme()
    g = sns.displot(as_map,
                    hue='student',
                    weights='count',
                    x='bucket',
                    kind='hist',
                    element='poly',
                    bins=int(number_of_days/2))

    ticks = list(range(0,number_of_days+1, int(number_of_days/_DAYS_PER_TICK)))
    g.set(xlim=(0,number_of_days + 1), xticks=ticks,
          xticklabels=[str(begin_time.date() + timedelta(days=x)) for x in ticks])
    g.set_xticklabels(rotation=30)
    g.tight_layout()
    g.set_axis_labels("Date", "Contributed")
    plt.show()


def _parse_explicit_line(explicit_line: str) -> tuple[str, list[str]]:
    tokens = explicit_line.replace(',', ' ').split()
    hash_id = tokens[0].strip().lower()
    authors = [name.strip().lower() for name in tokens[1:]]
    return (hash_id, authors)


def _read_explicit_map(explicit_path: str) -> dict[str, list[str]]:
    explicit_map = {}
    with open(explicit_path) as infile:
        for line in infile:
            if not line.strip():
                continue
            hash_id, authors = _parse_explicit_line(line)
            explicit_map[hash_id] = authors
    return explicit_map


def _extract_explicit_commits_from_repo(explicit: dict[str, list[str]],
                                        repo: Repository,
                                        configuration: _ProjectConfiguration) -> list[_CommitInfo]:
    commits = [_extract_contributions_from_commit(commit, configuration)
               for commit in repo.traverse_commits()
               if commit.hash in explicit]
    for commit in commits:
        if commit:
            commit.authors = explicit[commit.hash]
    return [commit for commit in commits if commit]


def _parse_time(date_string: str) -> datetime.datetime:
    return (datetime.datetime.strptime(date_string, '%Y-%m-%d')
                             .replace(tzinfo=datetime.timezone.utc))

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--begin',
                        help='Starting time in YYYY-MM-DD at midnight (Defaults to beginning of repo)', # noqa: E501
                        type=_parse_time)
    parser.add_argument('--end',
                        help='Ending time in YYY-MM-DD at midnight (Defaults to present time)',
                        default=datetime.datetime.now(datetime.timezone.utc),
                        type=_parse_time)
    parser.add_argument('--repo',
                        help='Location of git repository (Defaults to current directory)',
                        default=os.getcwd())
    parser.add_argument('--branch',
                        help='If only one branch should be included, the name of that branch (None by default)') # noqa: E501
    parser.add_argument('--projectkind',
                        help='Project kind used to determine file filters, quality checks, etc.',
                        choices=[key for key in _FILTERS if key])
    parser.add_argument('--pathexclusion',
                        help='Regular expression that matches paths of files that should be excluded, e.g. \'json|net\'') # noqa: E501
    parser.add_argument('--aliases',
                        help='Map some email addresses to others, e.g \'{"hmm@email.com": "hmm@sfu.ca"}\'', # noqa: E501
                        type=json.loads)
    parser.add_argument('--topk',
                        help='Given K, the K largest commits (in added code) are printed.',
                        default=15,
                        type=int)
    parser.add_argument('--stripdomains',
                        help='Strip email addresses to just user IDs',
                        default=True,
                        type=bool)
    parser.add_argument('--remotes',
                        help='Include remote repos.',
                        default=True,
                        type=bool)
    parser.add_argument('--maxsize',
                        help='Prune any commit that adds more than this number of lines.',
                        default=1000,
                        type=int)
    parser.add_argument('--explicitlist',
                        help='File list containing commits and authors for group work.',
                        default=None,
                        type=str)
    return parser


def _print_sized_commit(commit: Commit) -> None:
    print(' {:>6} {} {}'.format(commit.modified.lines_added,
                                commit.hash,
                                ', '.join(commit.authors)))


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO)
    logging.getLogger('pydriller.repository').disabled = True
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('git.cmd').disabled = True
    logging.getLogger('PIL.PngImagePlugin').disabled = True


PATH_EXCLUSION_FILE = '.pathexclusion'

def main() -> None:
    _configure_logging()

    parser = _build_arg_parser()
    args = parser.parse_args()

    begin_time     = args.begin
    end_time       = args.end
    repo_path      = args.repo
    branch         = args.branch
    topk           = args.topk
    max_size       = args.maxsize
    excluded_paths = args.pathexclusion
    strip_domains  = args.stripdomains
    aliases        = args.aliases
    explicit       = args.explicitlist
    remotes        = args.remotes
    if not aliases or not isinstance(aliases, dict):
        aliases = {}

    if os.path.exists(PATH_EXCLUSION_FILE):
        base = excluded_paths + '|' if excluded_paths else ''
        with open(PATH_EXCLUSION_FILE) as infile:
            excluded_paths = base + '|'.join(x.strip() for x in infile.readlines())
        logger.info('Loaded Exclusion: %s', excluded_paths)

    path_filter = re.compile(excluded_paths) if excluded_paths else None
    should_count = _FILTERS[args.projectkind]
    def count_filter(m: ModifiedFile) -> bool:
        return (should_count(m)
            and not (path_filter and m.new_path and path_filter.search(m.new_path)))
    configuration = _ProjectConfiguration(count_filter,
                                         lambda m: True,
                                         _count_nontrivial_changes_in_cpp)


    repo = Repository(repo_path, since=begin_time, to=end_time, only_in_branch=branch)

    if not explicit:
        repo = Repository(repo_path,
                          since=begin_time,
                          to=end_time,
                          only_in_branch=branch,
                          include_remotes=remotes)
        commits = _extract_commits_from_repo(repo, configuration)
    else:
        explicit_commits = _read_explicit_map(explicit)
        repo = Repository(repo_path,
                          since=begin_time,
                          to=end_time,
                          only_commits=list(explicit_commits.keys()),
                          include_remotes=True,
                          include_refs=True)
        commits = _extract_explicit_commits_from_repo(explicit_commits, repo, configuration)

    if max_size:
        oversized = [commit for commit in commits if commit.modified.lines_added >= max_size]
        commits = [commit for commit in commits if commit.modified.lines_added < max_size]
        if oversized:
            print('Skipping overly large commits')
            for commit in oversized:
                _print_sized_commit(commit)

    if strip_domains:
        for commit in commits:
            commit.authors = [author.split('@')[0] for author in commit.authors]

    per_student = _compute_per_student(commits, aliases)

    # Print out the student contributions ordered by ID
    print('\nStudent Contributions')
    print('=====================')
    for student in sorted(per_student.keys()):
        print(student, sum(_contributed_lines(commit) for commit in per_student[student]))

    # Print out the top K commits
    print('\nLargest Commits')
    print('=====================')
    largest = heapq.nlargest(topk, commits, key=lambda c: c.modified.lines_added)
    for commit in largest:
        _print_sized_commit(commit)

    # Plot out the contributions over time
    _plot_contributions(commits, begin_time, end_time)


if __name__ == '__main__':
    main()
