#!/bin/bash -l

set -euo pipefail

# Resolve paths relative to this script so it works from any cwd.
# Script lives at repo root; the Maven build root is the nested flink/ dir.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLINK_DIR="${REPO_ROOT}/flink"

CONFIG_PATH="${1:?usage: $0 <path-to-per-run-config.json>}"

# Flink 1.17 supports Java 8/11/17 (not 21). Default to the local JDK 17.
: "${JAVA_HOME:=/Users/nataliaw/Library/Java/JavaVirtualMachines/ms-17.0.18/Contents/Home}"

export JAVA_HOME
export PATH="$JAVA_HOME/bin:$PATH"

# Run from repo root so LocalProfile's default outputDir (user.dir/benchmark-results)
# lands in the repo-root benchmark-results/ dir. Flink is a 'provided' dependency (not
# in the fat jar), so we run one config on an embedded MiniCluster via Maven with the
# TEST classpath (includes provided deps) — no Flink install needed. This is the
# local-mode analogue of Spark's `spark-submit --master local[*]`. mvn runs against the
# nested flink/ pom but keeps cwd here (exec:java runs in-JVM, inheriting mvn's cwd).
cd "${REPO_ROOT}"

mvn -q -f "${FLINK_DIR}/pom.xml" test-compile \
    org.codehaus.mojo:exec-maven-plugin:3.1.0:java \
    -Dexec.mainClass=clustering.benchmark.BenchmarkRunner \
    -Dexec.args="--config ${CONFIG_PATH}" \
    -Dexec.classpathScope=test
