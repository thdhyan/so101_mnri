#!/usr/bin/env bash
# Resolve SO-101 arm roles -> current /dev/ttyACM* by USB serial number.
#
# Why this exists: /dev/ttyACM* order follows USB enumeration order, which
# changes whenever you replug, move to a different port, or change hubs. On
# 2026-09-25 moving to a different USB port silently SWAPPED the two arms:
#
#   serial      role       before     after
#   5AAF263743  leader     ttyACM1    ttyACM0
#   5AAF263600  follower   ttyACM0    ttyACM1
#
# so a config that hardcodes /dev/ttyACM1 as the leader became exactly
# backwards, and the recorder would have driven the wrong arm.
#
# Serials are the stable identity. This resolves them to whatever the ports
# happen to be right now, and fails loudly if a serial is missing rather than
# silently guessing.
#
# Usage:
#   source scripts/resolve_arm_ports.sh          # sets LEADER_PORT/FOLLOWER_PORT
#   bash scripts/resolve_arm_ports.sh --leader 5AAF263743 --follower 5AAF263600
#
# Emits shell assignments on stdout when given --emit, so it composes:
#   eval "$(bash scripts/resolve_arm_ports.sh --leader "$L" --follower "$F" --emit)"

set -euo pipefail

port_for_serial() {
    local want="$1" dev serial
    for dev in /dev/ttyACM*; do
        [ -e "$dev" ] || continue
        serial="$(udevadm info --query=property --name="$dev" 2>/dev/null \
                  | sed -n 's/^ID_SERIAL_SHORT=//p')"
        if [ "$serial" = "$want" ]; then
            echo "$dev"
            return 0
        fi
    done
    return 1
}

resolve() {
    local leader_ser="$1" follower_ser="$2" emit="$3" lp fp

    if [ "$leader_ser" = "$follower_ser" ]; then
        echo "ERROR: leader and follower serials are both '$leader_ser'" >&2
        return 1
    fi

    if ! lp="$(port_for_serial "$leader_ser")"; then
        echo "ERROR: no /dev/ttyACM* with serial '$leader_ser' (leader)." >&2
        echo "       Arms currently visible:" >&2
        for dev in /dev/ttyACM*; do
            [ -e "$dev" ] || continue
            echo "         $dev  serial=$(udevadm info --query=property --name="$dev" 2>/dev/null | sed -n 's/^ID_SERIAL_SHORT=//p')" >&2
        done
        return 1
    fi

    if ! fp="$(port_for_serial "$follower_ser")"; then
        echo "ERROR: no /dev/ttyACM* with serial '$follower_ser' (follower)." >&2
        return 1
    fi

    if [ "$lp" = "$fp" ]; then
        echo "ERROR: both roles resolved to $lp" >&2
        return 1
    fi

    if [ "$emit" = "--emit" ]; then
        echo "LEADER_PORT=$lp"
        echo "FOLLOWER_PORT=$fp"
    else
        echo "leader   serial $leader_ser  -> $lp"
        echo "follower serial $follower_ser  -> $fp"
    fi
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    LEADER_SER="${LEADER_SERIAL:-5AAF263743}"
    FOLLOWER_SER="${FOLLOWER_SERIAL:-5AAF263600}"
    while [ $# -gt 0 ]; do
        case "$1" in
            --leader)   LEADER_SER="$2";   shift 2 ;;
            --follower) FOLLOWER_SER="$2"; shift 2 ;;
            --emit)     EMIT="--emit";     shift ;;
            *) echo "unknown arg: $1" >&2; exit 2 ;;
        esac
    done
    resolve "$LEADER_SER" "$FOLLOWER_SER" "${EMIT:-}"
fi
