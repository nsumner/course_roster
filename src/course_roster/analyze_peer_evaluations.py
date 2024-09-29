#!/usr/bin/env python3

import argparse
import datetime
import json
import os
import random
import sys
import textwrap
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import ceil
from typing import Optional, TypeAlias

import matplotlib as mpl  # type: ignore[import-untyped]
import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import matplotlib.text as mpltext  # type: ignore[import-untyped]
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages  # type: ignore[import-untyped]
from tqdm import tqdm

from . import roster

# Optional dependencies require a bit more care
try:
    from wordcloud import WordCloud  # type: ignore[import-untyped]
except ImportError:
    _HAS_WORDCLOUD = False
else:
    _HAS_WORDCLOUD = True

try:
    from sumy.nlp.tokenizers import Tokenizer as SumyTokenizer  # type: ignore[import-not-found]
    from sumy.parsers.plaintext import PlaintextParser as SumyParser  # type: ignore[import-not-found]
    from sumy.summarizers.lsa import LsaSummarizer as SumySummarizer  # type: ignore[import-not-found]
except ImportError:
    _HAS_SUMY = False
else:
    _HAS_SUMY = True


mpl.rcParams.update({
    "text.usetex": "true",
    "font.family": "serif",
    "font.size": "6",
    "font.serif": [],
    "font.sans-serif": [],
})


# TODO: Separate the specifics of the peer evaluation into separate configuration

OVERALL = [
    'Work was divided fairly and reasonably.',
    'The team developed a design based on the known requirements of the project.',
    'The team developed goals based on the known requirements of the project.',
    'The team achieved the goals set out for this iteration.',
]
OVERALL_RANKS = 5


DIMENSIONS = [
    'technical',
    'productivity',
    'team work',
    'communication',
]
DIMENSION_RANKS = 3


WRITTEN = [
    ("wentwell",
     "What went well?"),
    ("wentpoorly",
     "What went poorly?"),
    ("othercomments",
     "Are there any other issues you have encountered or comments that you have so far?"),
]


@dataclass
class EvalConfig:
    overall: list[str]
    overall_ranks: int

    dimensions: list[str]
    dimension_ranks: int

    written: list[tuple[str, str]]


@dataclass
class AnalysisContext:
    results: pd.DataFrame
    group_name: str
    pdf: PdfPages
    config: EvalConfig


#############################################################################
# Analysis and visualization
#############################################################################


# LaTeX Helpers

# Because we are using LaTeX to render the figures, word wrapping gets ignored
# by matplotlib. This means that we need to perform wrapping via LaTeX ourselves.
# These helpers provide either a direct approach for labels or a drawable Text
# object for convenience.

def _wrapped_latex(text: str, width: str, centered: bool = False) -> str:
    centering = '\\centering' if centered else ''
    return f'\\parbox{{{width}}}{{{centering} {text}}}'


# `TexWrappedText` provides a class that can do this for a
# *provided* paragraph width. The downside is that this width must be manually
# chosen instead of inferred.
class TexWrappedText(mpltext.Text):  # type: ignore[misc]
    def __init__(self: mpltext.Text,  # type: ignore[no-untyped-def]
                 x: int = 0,
                 y: int = 0,
                 text: str = '',
                 width: str = '2cm',
                 **kwargs) -> None:
        mpltext.Text.__init__(self, x=x, y=y, text=text, wrap=True, **kwargs)
        self.width = width

    def _get_wrapped_text(self) -> str:
        return _wrapped_latex(self.get_text(), self.width)


# Histogramming

def generate_rank_colors(max_rank: int) -> list[tuple[float, float, float]]:
    mid = (1 + max_rank) // 2
    red = 1.0
    blue = 1.0
    return [
        (red * (1 - (i - 1.0) / (mid - 1.0)), 0.0, 0.0) for i in range(1, mid)
    ] + [
        (0.0, 0.0, blue * (i - mid) / (max_rank - mid)) for i in range(mid, max_rank + 1)
    ]


# A `BinNote` adds an annotation to a particular bin within a histogram.
# For example, (2, 'rx', 'Self') adds a red X with the label 'Self' on the
# x axis to the bin for the value 2.
BinNote: TypeAlias = tuple[float, str, str]


def construct_histogram(data: pd.DataFrame,
                        label: str,
                        figure: mpl.figure.Figure,
                        axis: mpl.axes.Axes,
                        max_rank: int,
                        title: str,
                        strip_y: bool = False,
                        markers: Optional[Sequence[BinNote]] = None) -> None:
    bar_margin = 0.6

    ax = sns.histplot(ax=axis, data=data, x=label, bins=np.arange(0.5, max_rank + 1, 1))

    small_font = 6
    ax.set_title(title, fontsize=small_font)
    ax.set_xlabel("Rating", fontsize=small_font)
    ax.set_ylabel("Frequency", fontsize=small_font)
    ax.set_xlim(1 - bar_margin, max_rank + bar_margin)
    ax.set_xticks(range(1, max_rank + 1))
    ax.set_ylim(bottom=0)
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.tick_params(labelsize=small_font)
    if strip_y:
        ax.axes.get_yaxis().set_visible(False)

    # We can use the positions of the xticks along the x axis to determine
    # where each patch is within the plot. That, in turn, can be used to
    # select the color of a patch.
    #
    # Drawing is necessary to make the positions of the graph elements active
    figure.canvas.draw()

    ticks = ax.transData.transform([(x, 0) for x in range(1, max_rank + 1)])
    palette = generate_rank_colors(max_rank)
    colors = list(zip((x for x, y in ticks), palette, strict=True))

    for bin in ax.patches:
        bounds = bin.get_window_extent()
        color = next(color for x, color in colors
                     if bounds.x0 <= x <= bounds.x1)
        bin.set_facecolor(color)

    if markers:
        # Heuristically toggle between left and right positioning
        # to decrease the likelyhood of collisions. For simple use cases
        # that should convey the information without complexity.
        # I do not presently expect more than 2 markers per histogram.
        position_toggle = True
        for bin, marker, marker_label in markers:
            ymin, ymax = ax.get_ylim()
            xy = (bin, ymin if position_toggle else ymax)
            ax.plot(xy[0], xy[1], marker, clip_on=False, mew=2)
            ax.annotate(marker_label,
                        xy=xy,
                        horizontalalignment='left' if position_toggle else 'right',
                        verticalalignment='top' if position_toggle else 'bottom')
            position_toggle = not position_toggle


def overall_histogram(df: pd.DataFrame,
                      container: mpl.figure.Figure,
                      config: EvalConfig) -> None:
    chars_per_title_line = 120
    min_rank = 1
    max_rank = config.overall_ranks
    axes = container.subplots(1, len(config.overall))

    for index, (axis, statement) in enumerate(zip(axes, config.overall, strict=True)):
        label = f'overall{index}'
        title_width = chars_per_title_line // len(config.overall)
        title = '\n'.join(textwrap.wrap(statement, width=title_width))

        # Filter out invalid data
        view = df[df[label].between(min_rank, max_rank)]
        construct_histogram(view, label, container, axis, max_rank, title)


def individual_histogram(df: pd.DataFrame,
                         container: mpl.figure.Figure,
                         config: EvalConfig,
                         compact: bool = False,
                         markers: Optional[Sequence[Optional[Sequence[BinNote]]]] = None) -> None:
    min_rank = 1
    max_rank = config.dimension_ranks
    num_dimensions = len(config.dimensions)
    if compact:
        spec = container.add_gridspec(nrows=1, ncols=num_dimensions, wspace=0, hspace=0)
        axes = [container.add_subplot(spec[0, col]) for col in range(num_dimensions)]
    else:
        axes = container.subplots(1, num_dimensions)

    if not markers:
        markers = [None for x in config.dimensions]

    strip_y = False
    for axis, label, marker in zip(axes, config.dimensions, markers, strict=True):
        # Filter out invalid data
        view = df[df[label].between(min_rank, max_rank)]
        construct_histogram(view, label, container, axis, max_rank, label, strip_y, marker)
        strip_y = strip_y or compact


# Wordclouds

def create_wordcloud(text: str, axis: mpl.axes.Axes) -> None:
    assert _HAS_WORDCLOUD

    wordcloud = WordCloud(max_words=30,
                          background_color="white",
                          width=400,
                          height=400).generate(text)
    axis.imshow(wordcloud, interpolation='bilinear')


# Text summaries

def create_text_summary(text: str, axis: mpl.axes.Axes, column_width: str) -> None:
    assert _HAS_SUMY

    sentence_count = 5
    parser = SumyParser.from_string(text, SumyTokenizer('english'))
    summarizer = SumySummarizer()
    summary_sentences = summarizer(parser.document, sentence_count)
    summary = ' '.join([str(sentence) for sentence in summary_sentences])

    axis.set(xlim=(0, 1), ylim=(0, 1))
    note = TexWrappedText(0, 1, summary, fontsize=6, width=column_width,
                          ha='left', va='top')
    axis.add_artist(note)


# Composing analysis

def quantitative_summary_page(context: AnalysisContext) -> None:
    fig = plt.figure(constrained_layout=True)
    fig.suptitle(f'Analysis for {context.group_name}', fontsize=12)
    (overall_fig, of_self_fig, of_others_fig) = fig.subfigures(nrows=3, ncols=1)

    overall_fig.suptitle("Overall project status assessment", fontsize=10)
    overall_histogram(context.results, overall_fig, context.config)

    # Collapse the self assessments
    self_frame = pd.DataFrame({
        dimension: context.results['self ' + dimension]
        for dimension in context.config.dimensions
    })
    of_self_fig.suptitle("Individual assessment of self", fontsize=10)
    individual_histogram(self_frame, of_self_fig, context.config)

    # Collapse the incoming assessments
    incoming_frame = pd.DataFrame({
        dimension: context.results['incoming ' + dimension].sum()
        for dimension in context.config.dimensions
    })
    of_others_fig.suptitle("Individual assessment of peers", fontsize=10)
    individual_histogram(incoming_frame, of_others_fig, context.config)


def _has_words(text: str) -> bool:
    return any(c.isalpha() for c in text)


LATEX_SANITIZING_TRANSLATION = str.maketrans({
  '#': '\\#',
})


def text_response_summary_page(context: AnalysisContext) -> None:
    texts = [(prompt, ' '.join(context.results[key].tolist()).strip())
             for key, prompt in context.config.written]

    figure = plt.figure()
    figure.suptitle(f'Summaries for {context.group_name}', fontsize=12)
    subfigures = figure.subfigures(nrows=1, ncols=len(texts))

    for (title, text), subfigure in zip(texts, subfigures, strict=True):
        chars_per_title_line = 80
        title_width = chars_per_title_line // len(texts)
        wrapped_title = '\n'.join(textwrap.wrap(title, width=title_width))

        cleaned = text.translate(LATEX_SANITIZING_TRANSLATION)

        subfigure.suptitle(wrapped_title, fontsize=10)
        axes = subfigure.subplots(2, 1)

        for ax in axes:
            ax.set_axis_off()

        if not _has_words(cleaned):
            continue

        if _HAS_WORDCLOUD:
            create_wordcloud(cleaned, axes[0])
        if _HAS_SUMY:
            available_cm_per_page = 12
            text_width = round(available_cm_per_page / len(texts), 2)
            create_text_summary(cleaned, axes[1], f'{text_width}cm')


def summarize_group(context: AnalysisContext) -> None:
    quantitative_summary_page(context)
    context.pdf.savefig()
    plt.close()

    text_response_summary_page(context)
    context.pdf.savefig()
    plt.close()


def analyze_individuals(context: AnalysisContext) -> None:
    def get_self_marker(student: pd.Series,
                        dimension: str) -> list[tuple[float, str, str]]:
        value = float(student['self ' + dimension])
        if value == 0:
            return []
        return [(value, 'yx', 'self')]

    def get_outgoing_marker(student: pd.Series,
                            dimension: str) -> list[tuple[float, str, str]]:
        values = [x for x in student['outgoing ' + dimension] if x != 0]
        if len(values) == 0:
            return []
        return [(sum(values) / len(values), 'c*', 'outward')]

    fig = plt.figure(constrained_layout=True)
    fig.suptitle(f'Peer evaluations for {context.group_name}', fontsize=12)

    subfigures = fig.subfigures(nrows=ceil(len(context.results.index) / 2), ncols=2)

    students = (row for index, row in context.results.iterrows())
    for student, subfigure in zip(students, subfigures.flatten(), strict=False):
        student_frame = pd.DataFrame({
            dimension: student['incoming ' + dimension]
            for dimension in context.config.dimensions
        })
        markers = [
            get_self_marker(student, dimension) + get_outgoing_marker(student, dimension)
            for dimension in context.config.dimensions
        ]
        subfigure.suptitle(student.name, fontsize=10)
        individual_histogram(student_frame, subfigure, context.config, True, markers)

    context.pdf.savefig()
    plt.close()


def analyze_group(context: AnalysisContext) -> None:
    summarize_group(context)
    analyze_individuals(context)


def analyze_data(results: pd.DataFrame,
                 students: roster.Roster,
                 group_label: str,
                 filename: str,
                 config: EvalConfig) -> None:

    with PdfPages(filename) as pdf:
        summarize_group(AnalysisContext(results, 'the whole class', pdf, config))

        groups = list(students.group_by(group_label))
        groups.sort()
        for group_name, group in tqdm(groups):
            group_results = results.loc[group[roster.Roster.Field.EMAIL.value]]
            analyze_group(AnalysisContext(group_results, group_name, pdf, config))

        d = pdf.infodict()
        d['Title'] = 'Peer Evaluation Results'
        d['Author'] = 'Nick Sumner'
        d['CreationDate'] = d['ModDate'] = datetime.datetime.now()


#############################################################################
# Evaluation data ingestion
#############################################################################

JSON: TypeAlias = Mapping[str, "JSON"] | Sequence["JSON"] | str | int | float | bool | None


def create_initial_dataframe(student_ids: list[str],
                             config: EvalConfig) -> pd.DataFrame:
    empty_strings = ['' for student in student_ids]
    zeros = [0 for student in student_ids]

    initial_data: dict[str, JSON] = {
        'username': student_ids,
    }
    initial_data.update({
        f'overall{count}': zeros
        for count in range(len(config.overall))
    })
    initial_data.update({
        f'self {dimension}': zeros
        for dimension in config.dimensions
    })
    initial_data.update({
        f'incoming {dimension}': [[] for student in student_ids]
        for dimension in config.dimensions
    })
    initial_data.update({
        f'outgoing {dimension}': [[] for student in student_ids]
        for dimension in config.dimensions
    })
    initial_data.update({
        label: empty_strings
        for label, prompt in config.written
    })

    table = pd.DataFrame(initial_data)
    table.set_index('username', inplace=True, drop=True)

    return table


def collect_student_data(evaluation_dir: str,
                         student_ids: list[str],
                         filename: str,
                         config: EvalConfig) -> pd.DataFrame:
    start_dir = os.getcwd()

    results = create_initial_dataframe(student_ids, config)

    for student in student_ids:
        json_path = os.path.join(evaluation_dir, student, filename)

        if not os.path.exists(json_path) or not os.access(json_path, os.R_OK):
            print('Unable to read:', json_path, file=sys.stderr)
            continue

        with open(json_path) as infile:
            response = json.load(infile)

        match response:
            case {
                    'student':    found_student,
                    'overall':    overall,
                    'individual': individual,
                    'written':    written,
            }:
                pass
            case _:
                print("Invalid json content for evaluation.")
                continue

        if student != found_student:
            print(f'Student and username do not match: {student}, {found_student}')

        # individual = response['individual']
        # overall = response['overall']

        # TODO: Ensure that student's response covers exactly those people
        #  in the student's group, no more and no less.
        if student not in individual:
            print('Self evaluation not present', file=sys.stderr)
            continue

        # NOTE: Storing lists and objects in a DataFrame gets more complicated
        # that setting standard value types. Thus, we collect all of the normal
        # value types and set them for the row together before setting list
        # contents one at a time.

        written_keys = [label for label, prompt in config.written]
        written_values = [written[key] for key in written_keys]

        num_overall = len(config.overall)
        overall_keys = [f'overall{count}' for count in range(num_overall)]
        overall_values = [int(overall[str(count)]) for count in range(num_overall)]

        self_keys = [f'self {dimension}'
                     for dimension in config.dimensions]
        self_values = [int(individual[student][dimension])
                       for dimension in config.dimensions]

        combined_keys = written_keys + overall_keys + self_keys
        combined_values = written_values + overall_values + self_values
        results.loc[student, combined_keys] = combined_values

        # All other evaluations are of peers, so we can remove the self eval.
        # This removes the self cornercase when collecting the information
        # about reviewing others below.
        individual.pop(student, None)

        for dimension in config.dimensions:
            key = f'outgoing {dimension}'
            value_list = [int(peer_eval[dimension])
                          for peer, peer_eval in individual.items()]
            results.at[student, key] = value_list

        for peer, peer_eval in individual.items():
            for dimension in config.dimensions:
                key = f'incoming {dimension}'
                value = int(peer_eval[dimension])
                results.loc[peer, key].append(value)

    os.chdir(start_dir)

    return results


#############################################################################
# Test data generation
#############################################################################

FREQUENT_WORDS = [
    'add', 'air', 'animal', 'answer', 'area', 'base', 'beauty', 'began', 'big',
    'bird', 'black', 'blue', 'boat', 'body', 'book', 'box', 'boy', 'bring',
    'brought', 'build', 'busy', 'car', 'care', 'carry', 'center', 'change',
    'check', 'children', 'city', 'class', 'clear', 'close', 'cold', 'color',
    'common', 'complete', 'correct', 'country', 'cover', 'cross', 'cut', 'dark',
    'day', 'decide', 'deep', 'develop', 'differ', 'direct', 'distant', 'dog',
    'door', 'draw', 'drive', 'dry', 'early', 'earth', 'ease', 'east', 'eat',
    'equate', 'eye', 'face', 'fact', 'fall', 'family', 'farm', 'fast', 'father',
    'feel', 'feet', 'field', 'figure', 'final', 'fine', 'fish', 'fly', 'follow',
    'food', 'foot', 'force', 'form', 'free', 'friend', 'game', 'girl', 'gold',
    'good', 'govern', 'great', 'green', 'ground', 'group', 'grow', 'hand',
    'happen', 'hard', 'head', 'hear', 'heard', 'heat', 'high', 'hold', 'horse',
    'hot', 'hour', 'house', 'idea', 'inch', 'island', 'kind', 'king', 'knew',
    'land', 'language', 'large', 'late', 'laugh', 'lay', 'lead', 'learn',
    'leave', 'left', 'letter', 'life', 'light', 'list', 'listen', 'live',
    'long', 'love', 'machine', 'main', 'man', 'map', 'mark', 'measure', 'men',
    'mile', 'mind', 'minute', 'money', 'moon', 'morning', 'mother', 'mountain',
    'multiply', 'music', 'night', 'north', 'note', 'notice', 'noun', 'number',
    'numeral', 'object', 'ocean', 'open', 'order', 'paint', 'paper', 'pass',
    'pattern', 'people', 'person', 'picture', 'piece', 'place', 'plain', 'plan',
    'plane', 'plant', 'play', 'point', 'port', 'pose', 'pound', 'power',
    'press', 'problem', 'produce', 'product', 'pull', 'question', 'quick',
    'rain', 'reach', 'read', 'ready', 'real', 'record', 'red', 'remember',
    'rest', 'river', 'road', 'rock', 'room', 'rule', 'school', 'science', 'sea',
    'sentence', 'serve', 'set', 'shape', 'ship', 'short', 'simple', 'sing',
    'slow', 'small', 'snow', 'song', 'sound', 'south', 'space', 'special',
    'spell', 'stand', 'star', 'start', 'stay', 'stead', 'step', 'stood',
    'story', 'street', 'strong', 'study', 'sun', 'surface', 'table', 'tail',
    'talk', 'teach', 'test', 'thought', 'time', 'tire', 'told', 'town',
    'travel', 'tree', 'true', 'turn', 'unit', 'usual', 'verb', 'voice', 'vowel',
    'wait', 'walk', 'war', 'warm', 'watch', 'water', 'week', 'west', 'wheel',
    'white', 'wind', 'wood', 'word', 'work', 'write', 'year', 'young',
]


def generate_text() -> str:
    num_pseudosentences = 5
    sentences = [' '.join(random.sample(FREQUENT_WORDS, k=10)) + '.'
                 for i in range(num_pseudosentences)]
    return ' '.join(sentences)


def generate_evaluation_json(student: str,
                             usernames: list[str],
                             config: EvalConfig) -> str:
    evaluation = {
        'student': student,
        'overall': {
            str(statement): random.randrange(1, config.overall_ranks + 1)
            for statement in range(len(config.overall))
        },
        'individual': {
            peer: {
                dimension: random.randrange(1, config.dimension_ranks + 1)
                for dimension in config.dimensions
            }
            for peer in usernames
        },
        'written': {
            label: generate_text()
            for label, question in config.written
        },
    }
    return json.dumps(evaluation)


def generate_data(directory: str,
                  roster_csv: str,
                  group_label: str,
                  filename: str,
                  config: EvalConfig) -> None:
    if not os.path.exists(roster_csv):
        class_size = 20
        students = roster.from_nothing(class_size)
    else:
        students = roster.from_roster_csv(roster_csv)

    if group_label not in students.table.columns:
        group_size = 8
        num_groups = int(ceil(len(students.table.index) / group_size))
        group_names = [f'Team {chr(65 + i)}' for i in range(num_groups)]
        roster.group_students_randomly(students,
                                       group_size,
                                       group_label,
                                       group_names)

    roster.to_roster_csv(students, roster_csv)

    for _, group in students.group_by(group_label):
        usernames = group[roster.Roster.Field.EMAIL.value].values.tolist()
        for student in usernames:
            student_dir = os.path.join(directory, student)
            if not os.path.exists(student_dir):
                os.mkdir(student_dir)

            student_eval_path = os.path.join(student_dir, filename)
            if not os.path.exists(student_eval_path):
                eval_json = generate_evaluation_json(student, usernames, config)
                with open(student_eval_path, 'w') as outfile:
                    outfile.write(eval_json)


#############################################################################
# Primary entrypoint and validation
#############################################################################


def validate_options(args: argparse.Namespace) -> None:
    if not os.path.exists(args.roster_csv):
        raise ValueError('Roster not found. '
                         'Check that you provided a correct --roster-csv')

    if not os.path.exists(args.evaluation_dir):
        raise ValueError('Submission directory does not exist. '
                         'Check --evaluation-dir.')

    if not os.listdir(args.evaluation_dir):
        raise ValueError('Submission directory is empty.'
                         'Check --evaluation-dir.')


def check_for_submissions(students: list[str],
                          evaluation_dir: str) -> None:
    student_set = set(students)
    subdirectories = {x for x in os.listdir(evaluation_dir)
                      if os.path.isdir(os.path.join(evaluation_dir, x))}

    submissions = list(student_set.intersection(subdirectories))
    if len(submissions) == 0:
        raise ValueError('No evaluations found in --evaluation-dir')

    extra_directories = subdirectories - student_set
    if len(extra_directories) != 0:
        print('Extra subdirectories found:\n\t', '\n\t'.join(extra_directories), file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--evaluation-dir',
                        help='Directory containing unpacked evaluations from each student.',
                        default=os.getcwd())
    parser.add_argument('--roster-csv',
                        help='Path to roster file containing group information.',
                        default='roster.csv')
    parser.add_argument('--group-label',
                        help='Column name for groups in the roster CSV.',
                        default='Team')
    parser.add_argument('--filename',
                        help='Consistent filename for student submissions.',
                        default='evaluation.json')
    parser.add_argument('--output',
                        help='Filename for resulting PDF analysis.',
                        default='peer-evaluation-results.pdf')
    parser.add_argument('--generate-test-data',
                        help='Generate and analyze fake data as specified.',
                        action='store_true')

    args = parser.parse_args()

    config = EvalConfig(OVERALL, OVERALL_RANKS, DIMENSIONS, DIMENSION_RANKS, WRITTEN)

    if args.generate_test_data:
        generate_data(args.evaluation_dir,
                      args.roster_csv,
                      args.group_label,
                      args.filename,
                      config)

    validate_options(args)

    students = roster.from_roster_csv(args.roster_csv)
    usernames = students.get_email_ids()
    usernames.sort()

    check_for_submissions(usernames, args.evaluation_dir)

    evaluation_data = collect_student_data(args.evaluation_dir, usernames, args.filename, config)

    analyze_data(evaluation_data, students, args.group_label, args.output, config)


if __name__ == '__main__':
    main()
