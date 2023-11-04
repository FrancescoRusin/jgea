package io.github.ericmedvet.jgea.experimenter.listener.decoupled;

import io.github.ericmedvet.jgea.core.util.HashMapTable;
import io.github.ericmedvet.jgea.core.util.Pair;
import io.github.ericmedvet.jgea.core.util.Table;

import java.time.LocalDateTime;
import java.util.Collection;
import java.util.Map;

/**
 * @author "Eric Medvet" on 2023/11/03 for jgea
 */
public class TabledSource {

  public static final String VALUE_NAME = "value";
  private final Source source;

  private final int lastN;
  protected final Table<Pair<LocalDateTime, MachineKey>, String, MachineInfo> machines;
  protected final Table<Pair<LocalDateTime, ProcessKey>, String, ProcessInfo> processes;
  protected final Table<Pair<LocalDateTime, ProcessKey>, String, LogInfo> logs;
  protected final Table<Pair<LocalDateTime, ExperimentKey>, String, ExperimentInfo> experiments;
  protected final Table<Pair<LocalDateTime, RunKey>, String, RunInfo> runs;
  protected final Table<Pair<LocalDateTime, DataItemKey>, String, DataItemInfo> dataItems;

  private LocalDateTime lastRefresh;

  public TabledSource(Source source, int lastN) {
    this.source = source;
    this.lastN = lastN;
    machines = new HashMapTable<>();
    processes = new HashMapTable<>();
    logs = new HashMapTable<>();
    experiments = new HashMapTable<>();
    runs = new HashMapTable<>();
    dataItems = new HashMapTable<>();
    lastRefresh = LocalDateTime.MIN;
  }

  private static <K> void prune(Table<Pair<LocalDateTime, K>, String, ?> table, int n) {
    Collection<K> ks = table.rowIndexes().stream().map(Pair::second).distinct().toList();
    ks.forEach(k -> table.rowIndexes().stream()
        .filter(p -> p.second().equals(k))
        .sorted((p1, p2) -> p2.first().compareTo(p1.first()))
        .skip(n)
        .forEach(table::removeRow));
  }

  public Table<Pair<LocalDateTime, DataItemKey>, String, DataItemInfo> getDataItems() {
    return dataItems;
  }

  public Table<Pair<LocalDateTime, ExperimentKey>, String, ExperimentInfo> getExperiments() {
    return experiments;
  }

  public Table<Pair<LocalDateTime, ProcessKey>, String, LogInfo> getLogs() {
    return logs;
  }

  public Table<Pair<LocalDateTime, MachineKey>, String, MachineInfo> getMachines() {
    return machines;
  }

  public Table<Pair<LocalDateTime, ProcessKey>, String, ProcessInfo> getProcesses() {
    return processes;
  }

  public Table<Pair<LocalDateTime, RunKey>, String, RunInfo> getRuns() {
    return runs;
  }

  public void refresh() {
    source.machineKeys().forEach(mk -> source.machineInfos(mk, lastRefresh)
        .forEach((t, v) -> machines.set(new Pair<>(t, mk), VALUE_NAME, v
        )));
    source.processKeys().forEach(pk -> source.processInfos(pk, lastRefresh)
        .forEach((t, v) -> processes.set(new Pair<>(t, pk), VALUE_NAME, v
        )));
    source.processKeys().forEach(pk -> source.logInfos(pk, lastRefresh)
        .forEach((t, v) -> logs.set(new Pair<>(t, pk), VALUE_NAME, v)));
    source.experimentKeys().forEach(ek -> source.experimentInfos(ek, lastRefresh)
        .forEach((t, v) -> experiments.addRow(new Pair<>(t, ek), Map.ofEntries(
            Map.entry(VALUE_NAME, v)
        ))));
    source.runKeys().forEach(rk -> source.runInfos(rk, lastRefresh)
        .forEach((t, v) -> runs.set(new Pair<>(t, rk), VALUE_NAME, v)));
    source.dataItemKeys().forEach(dik ->
        source.dataItemInfos(dik, lastRefresh)
            .forEach((t, v) -> dataItems.set(new Pair<>(t, dik), VALUE_NAME, v)));
    lastRefresh = LocalDateTime.now();
    if (lastN > 0) {
      prune(machines, lastN);
      prune(processes, lastN);
      prune(logs, lastN);
      prune(experiments, lastN);
      prune(runs, lastN);
      prune(dataItems, lastN);
    }
  }
}
