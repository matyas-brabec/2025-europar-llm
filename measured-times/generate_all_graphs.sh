#!/bin/bash

script_dir=$(dirname "$0")
output_dir=$script_dir/graphs

cd "$output_dir" || exit 1

printer_path="$script_dir/../../framework/print-graph.py"

# All Graphs
args_per_graph=(

    # normal scale
    "hist@single-graph-style@no-paper-print@auto-boundaries ../histogram-hexdump-ampere.csv ../histogram-hexdump-ampere.reference.csv"
    "hist@single-graph-style@no-paper-print@auto-boundaries ../histogram-hexdump-volta.csv ../histogram-hexdump-volta.reference.csv"
    "hist@single-graph-style@no-paper-print@auto-boundaries ../histogram-hexdump-hopper.csv ../histogram-hexdump-hopper.reference.csv"

    "hist@single-graph-style@no-paper-print@auto-boundaries ../histogram-loremipsum-ampere.csv ../histogram-loremipsum-ampere.reference.csv"
    "hist@single-graph-style@no-paper-print@auto-boundaries ../histogram-loremipsum-volta.csv ../histogram-loremipsum-volta.reference.csv"
    "hist@single-graph-style@no-paper-print@auto-boundaries ../histogram-loremipsum-hopper.csv ../histogram-loremipsum-hopper.reference.csv"

    "gol@single-graph-style@no-paper-print@auto-boundaries ../gol-hopper.csv ../gol-hopper.reference.csv"
    "gol@single-graph-style@no-paper-print@auto-boundaries ../gol-ampere.csv ../gol-ampere.reference.csv"
    "gol@single-graph-style@no-paper-print@auto-boundaries ../gol-volta.csv ../gol-volta.reference.csv"

    # logaritmic scale
    "hist@single-graph-style@no-paper-print@auto-boundaries@with-log-scale ../histogram-hexdump-ampere.csv ../histogram-hexdump-ampere.reference.csv log-scale_"
    "hist@single-graph-style@no-paper-print@auto-boundaries@with-log-scale ../histogram-hexdump-volta.csv ../histogram-hexdump-volta.reference.csv log-scale_"
    "hist@single-graph-style@no-paper-print@auto-boundaries@with-log-scale ../histogram-hexdump-hopper.csv ../histogram-hexdump-hopper.reference.csv log-scale_"

    "hist@single-graph-style@no-paper-print@auto-boundaries@with-log-scale ../histogram-loremipsum-ampere.csv ../histogram-loremipsum-ampere.reference.csv log-scale_"
    "hist@single-graph-style@no-paper-print@auto-boundaries@with-log-scale ../histogram-loremipsum-volta.csv ../histogram-loremipsum-volta.reference.csv log-scale_"
    "hist@single-graph-style@no-paper-print@auto-boundaries@with-log-scale ../histogram-loremipsum-hopper.csv ../histogram-loremipsum-hopper.reference.csv log-scale_"

    "gol@single-graph-style@no-paper-print@auto-boundaries@with-log-scale ../gol-hopper.csv ../gol-hopper.reference.csv log-scale_"
    "gol@single-graph-style@no-paper-print@auto-boundaries@with-log-scale ../gol-ampere.csv ../gol-ampere.reference.csv log-scale_"
    "gol@single-graph-style@no-paper-print@auto-boundaries@with-log-scale ../gol-volta.csv ../gol-volta.reference.csv log-scale_"

    # Graphs for the paper

    "hist@two-graphs-in-row-style@first-graph@with-log-scale ../histogram-loremipsum-hopper.csv ../histogram-loremipsum-hopper.reference.csv paper_"
    "hist@two-graphs-in-row-style@second-graph@with-log-scale ../histogram-hexdump-hopper.csv ../histogram-hexdump-hopper.reference.csv paper_"
    "gol@single-graph-style@with-log-scale ../gol-hopper.csv ../gol-hopper.reference.csv paper_"

    "hist@two-graphs-in-row-style@first-graph@with-log-scale@narrow-hist ../histogram-loremipsum-hopper.csv ../histogram-loremipsum-hopper.reference.csv paper_narrow_"
    "hist@two-graphs-in-row-style@second-graph@with-log-scale@narrow-hist ../histogram-hexdump-hopper.csv ../histogram-hexdump-hopper.reference.csv paper_narrow_"
    "gol@single-graph-style@with-log-scale@narrow-gol ../gol-hopper.csv ../gol-hopper.reference.csv paper_narrow_"
)

echo "Printing graphs"
echo

(
    echo "Setting up Python environment"
    cd "$(dirname "$printer_path")" || exit 1

    python3 -m venv .venv
    source .venv/bin/activate
    python3 -m pip install --upgrade pip
    python3 -m pip install -r requirements.txt
)

for args in "${args_per_graph[@]}"; do
    echo "python3 $printer_path $args"
    sh -c "python3 $printer_path $args"
    echo
done
