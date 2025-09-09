#!/bin/bash

# -------------------------
# Settings
# -------------------------
JOB_INPUT_CSV_FILE="./auto_eval_jobs.csv"

POLICIES="Mlp Act Sarnn DiffusionPolicy"

POLICY_WALLTIME_Mlp="01:30:00"
POLICY_WALLTIME_Act="03:00:00"
POLICY_WALLTIME_Sarnn="08:00:00"
POLICY_WALLTIME_DiffusionPolicy="72:00:00"

MAX_JOBS=200
MAX_MINUTES=20160  # in minutes (336 hours)

JOB_SUBMIT_COOLDOWN=10

# Wait interval (seconds) used in the waiting loop
WAIT_SLEEP_SEC=900 # in seconds (15 minutes)

# Clean system queue directory
\rm -fvr ./.sys_queue_AutoEval/* || true

mkdir -pv ./result/ || true
cp -uv ~/proj/RoboManipBaselines-Eval/doc/evaluation_results.md ./result/ || true

# QSTAT user
QSTAT_USER="${QSTAT_USER:-$(whoami)}"

# -------------------------
# Functions
# -------------------------
# get_job_count: count lines that look like job lines (start with digit)
get_job_count() {
  qstat -u "$QSTAT_USER" 2>/dev/null | awk '$1 ~ /^[0-9]/ {count++} END {print (count+0)}'
}

# Convert HH:MM:SS to minutes (round seconds)
hms_to_minutes() {
  time_str=$1
  echo "$time_str" | awk -F: '{
    h = ($1==""?0:$1) + 0
    m = ($2==""?0:$2) + 0
    s = ($3==""?0:$3) + 0
    print h * 60 + m + int((s + 30) / 60)
  }'
}

# get_wall_minutes: build policy->walltime map from POLICIES and use it in awk
get_wall_minutes() {
  qstat_out=$(qstat -u "$QSTAT_USER" 2>/dev/null)
  [ -z "$qstat_out" ] && echo 0 && return

  # locate header line containing jobname/name (case-insensitive)
  header_line_no=$(echo "$qstat_out" | awk 'BEGIN{IGNORECASE=1} /jobname|job name|name|job id/ {print NR; exit}')
  if [ -z "$header_line_no" ]; then
    header_line_no=$(echo "$qstat_out" | nl -ba | awk '$2!~/^$/ {print $1; exit}')
  fi

  header=$(echo "$qstat_out" | sed -n "${header_line_no}p")
  [ -z "$header" ] && echo 0 && return

  # compute character start position of "Jobname" or "Name"
  start_pos=$(echo "$header" | awk '{
    s=tolower($0)
    if (match(s,"jobname")) {print RSTART; exit}
    if (match(s,"job name")) {print RSTART; exit}
    if (match(s,"name")) {print RSTART; exit}
    if (match(s,"job id")) {print RSTART; exit}
    print 0
  }')
  [ -z "$start_pos" ] || [ "$start_pos" -eq 0 ] && echo 0 && return

  # Build map string from POLICIES list: "Sarnn=02:00:00|Mlp=00:15:00|..."
  MAP=""
  for p in $POLICIES; do
    # get variable POLICY_WALLTIME_<Policy>
    wt_var="POLICY_WALLTIME_${p}"
    wt_val=$(eval "echo \${$wt_var:-}")
    if [ -n "$wt_val" ]; then
      if [ -z "$MAP" ]; then
        MAP="${p}=${wt_val}"
      else
        MAP="${MAP}|${p}=${wt_val}"
      fi
    fi
  done

  # pass start_pos and map to awk; awk will parse the map into an array
  echo "$qstat_out" | awk -v sp="$start_pos" -v map="$MAP" '
    BEGIN{
      # parse map "A=00:10:00|B=01:00:00" into arr[policy]=walltime
      n = split(map, pairs, /\|/)
      for (i=1; i<=n; ++i) {
        if (pairs[i] == "") continue
        m = split(pairs[i], kv, "=")
        if (m >= 2) arr[kv[1]] = kv[2]
      }
    }
    function hms_to_min(t,    a,h,m,s) {
      split(t,a,":")
      h=(a[1]+0); m=(a[2]+0); s=(a[3]+0)
      return h*60 + m + int((s+30)/60)
    }
    $0 ~ /./ {
      if ($1 !~ /^[0-9]/) next
      s = substr($0, sp)
      sub(/^[[:space:]]+/, "", s)
      n = split(s, arrtok, /[[:space:]]+/)
      name = arrtok[1]
      gsub(/[*]+$/, "", name)
      split(name, pparts, "_")
      if (length(pparts) > 1) policy = pparts[1]
      else if (match(name, /^[A-Za-z]+/)) policy = substr(name, RSTART, RLENGTH)
      else policy = name
      if (policy in arr) total += hms_to_min(arr[policy])
    }
    END { print (total+0) }
  '
}

sanitize_name() {
  raw="$1"
  safe=$(printf '%s' "$raw" | sed 's/[^A-Za-z0-9_.-]/_/g')
  safe=$(printf '%s' "$safe" | cut -c1-64)
  printf '%s' "$safe"
}

# -------------------------
# Main loop: read CSV
# -------------------------
tail -n +2 "$JOB_INPUT_CSV_FILE" | while IFS=',' read -r POLICY SEED ENV_NAME DATA_TAG SETTING_REMARK ARGS_FILE; do
  # skip empty or malformed lines
  [[ -z "$POLICY" ]] && continue

  WALLTIME_VAR="POLICY_WALLTIME_$POLICY"
  WALLTIME=$(eval "echo \$$WALLTIME_VAR")

  if [ -z "$WALLTIME" ]; then
    printf 'Error: Unknown policy %s (no walltime defined)\n' "$POLICY" >&2
    exit 1
  fi

  NEW_MINUTES=$(hms_to_minutes "$WALLTIME")

  if [ -z "$DATA_TAG" ]; then
    printf 'Error: Missing env_tag for policy=%s env=%s\n' "$POLICY" "$ENV_NAME" >&2
    exit 1
  fi

  # PBS queue selection
  case "$ENV_NAME" in
    DummyPrimaryEnv)
      PBS_QUEUE="rt_HF" ;;
    *)
      PBS_QUEUE="rt_HG" ;;
  esac

  RAW_JQUEUE_NAME="${POLICY}_${ENV_NAME}_seed${SEED}_${SETTING_REMARK}"
  JQUEUE_NAME=$(sanitize_name "$RAW_JQUEUE_NAME")

  # Wait until job slots and walltime budget are available
  while :; do
    CUR_JOBS=$(get_job_count)
    CUR_MINUTES=$(get_wall_minutes)
    TOTAL_MINUTES=$(( CUR_MINUTES + NEW_MINUTES ))

    if [ "$CUR_JOBS" -lt "$MAX_JOBS" ] && [ "$TOTAL_MINUTES" -le "$MAX_MINUTES" ]; then
      break
    fi

    echo "[wait] jobs=$CUR_JOBS/$MAX_JOBS, walltime=$CUR_MINUTES/$MAX_MINUTES, this_job_walltime=$NEW_MINUTES min -> sleep $WAIT_SLEEP_SEC s"
    sleep "$WAIT_SLEEP_SEC"
  done

  # Submit job
  # remove CR (\r) which often comes from Windows line endings
  ARGS_FILE="${ARGS_FILE//$'\r'/}"
  # remove surrounding double quotes if any (CSV exported with quotes)
  ARGS_FILE="${ARGS_FILE%\"}"
  ARGS_FILE="${ARGS_FILE#\"}"
  # trim leading/trailing whitespace (POSIX-safe)
  ARGS_FILE="$(printf '%s' "$ARGS_FILE" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
  if [ -z "$ARGS_FILE" ]; then
      ARGS_FILE_TRAIN=""
  else
      ARGS_FILE_TRAIN="args_train_configs/$ARGS_FILE"
  fi
  echo "[submit job] POLICY=$POLICY, ENV=$ENV_NAME, TAG=$DATA_TAG, SEED=$SEED, CAM=$SETTING_REMARK, QUEUE=$PBS_QUEUE"
  qsub -N "$JQUEUE_NAME" \
        -q "$PBS_QUEUE" -k oed -l walltime="${WALLTIME}" \
        -v POLICY="$POLICY",ENV="$ENV_NAME",JQUEUE="$JQUEUE_NAME",DATA_TAG="$DATA_TAG",SEEDS="$SEED",ARGS_FILE_TRAIN="$ARGS_FILE_TRAIN",SETTING_REMARK="$SETTING_REMARK" \
        ./eval_report_worker.sh.sh

  sleep "$JOB_SUBMIT_COOLDOWN"

  POST_QSTAT_JOBS=$(get_job_count)
  POST_QSTAT_MINUTES=$(get_wall_minutes)

  # Estimated values including the submitted job
  EST_JOBS=$(( POST_QSTAT_JOBS + 1 ))
  EST_MINUTES=$(( POST_QSTAT_MINUTES + NEW_MINUTES ))

  printf '[status-after-submit] qstat_report: jobs=%s/%s, walltime=%s/%s min.\n' \
    "$POST_QSTAT_JOBS" "$MAX_JOBS" "$POST_QSTAT_MINUTES" "$MAX_MINUTES"
  printf '[status-after-submit] est_this_job: jobs=%s/%s, walltime=%s/%s min (this_job=%s min)\n' \
    "$EST_JOBS" "$MAX_JOBS" "$EST_MINUTES" "$MAX_MINUTES" "$NEW_MINUTES"
done

exit 0
