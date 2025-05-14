## Package `dynamicalSystem.biAgentTask`

Aliases: `ds.baTask`, `ds.bat`, `ds.biAgentTask`, `dynSys.baTask`, `dynSys.bat`, `dynSys.biAgentTask`, `dynamicalSystem.baTask`, `dynamicalSystem.bat`, `dynamicalSystem.biAgentTask`

### Builder `dynamicalSystem.biAgentTask.fromEnvironment()`

`ds.bat.fromEnvironment(name; environment; stopCondition; tRange; dT)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `name` | s | interpolate `{environment.name}[{tRange.min};{tRange.max}]` | <code><abbr title="java.lang.String">String</abbr></code> |
| `environment` | npm |  | <code><abbr title="io.github.ericmedvet.jsdynsym.control.HomogeneousBiEnvironment">HomogeneousBiEnvironment</abbr>&lt;O, A, S, C&gt;</code> |
| `stopCondition` | npm |  | <code><abbr title="java.util.function.Predicate">Predicate</abbr>&lt;S&gt;</code> |
| `tRange` | npm |  | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |
| `dT` | d |  | <code>double</code> |

Produces <code><abbr title="io.github.ericmedvet.jsdynsym.control.HomogeneousBiAgentTask">HomogeneousBiAgentTask</abbr>&lt;C, O, A, S&gt;</code>; built from `io.github.ericmedvet.jsdynsym.buildable.builders.HomogeneousBiAgentTasks.fromEnvironment()` by jgea-experimenter:2.7.1-SNAPSHOT

## Package `dynamicalSystem.drawer`

Aliases: `ds.d`, `ds.drawer`, `dynSys.d`, `dynSys.drawer`, `dynamicalSystem.d`, `dynamicalSystem.drawer`

### Builder `dynamicalSystem.drawer.navigation()`

`ds.d.navigation()`

Produces <code><abbr title="io.github.ericmedvet.jsdynsym.control.navigation.NavigationDrawer">NavigationDrawer</abbr></code>; built from `io.github.ericmedvet.jsdynsym.buildable.builders.Drawers.navigation()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `dynamicalSystem.drawer.pointNavigation()`

`ds.d.pointNavigation()`

Produces <code><abbr title="io.github.ericmedvet.jsdynsym.control.navigation.PointNavigationDrawer">PointNavigationDrawer</abbr></code>; built from `io.github.ericmedvet.jsdynsym.buildable.builders.Drawers.pointNavigation()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `dynamicalSystem.drawer.pong()`

`ds.d.pong()`

Produces <code><abbr title="io.github.ericmedvet.jsdynsym.control.pong.PongDrawer">PongDrawer</abbr></code>; built from `io.github.ericmedvet.jsdynsym.buildable.builders.Drawers.pong()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `dynamicalSystem.drawer.vectorField()`

`ds.d.vectorField(arena)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `arena` | e |  | <code><abbr title="io.github.ericmedvet.jsdynsym.control.navigation.Arena$Prepared">Arena$Prepared</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jsdynsym.control.navigation.VectorFieldDrawer">VectorFieldDrawer</abbr></code>; built from `io.github.ericmedvet.jsdynsym.buildable.builders.Drawers.vectorField()` by jgea-experimenter:2.7.1-SNAPSHOT

## Package `dynamicalSystem.environment`

Aliases: `ds.e`, `ds.env`, `ds.environment`, `dynSys.e`, `dynSys.env`, `dynSys.environment`, `dynamicalSystem.e`, `dynamicalSystem.env`, `dynamicalSystem.environment`

### Builder `dynamicalSystem.environment.navigation()`

`ds.e.navigation(name; initialRobotXRange; initialRobotYRange; initialRobotDirectionRange; targetXRange; targetYRange; robotRadius; robotMaxV; sensorsAngleRange; nOfSensors; sensorRange; senseTarget; arena; rescaleInput; randomGenerator)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `name` | s | interpolate `nav-{arena}` | <code><abbr title="java.lang.String">String</abbr></code> |
| `initialRobotXRange` | npm | `m.range(min = 0.45; max = 0.55)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |
| `initialRobotYRange` | npm | `m.range(min = 0.8; max = 0.85)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |
| `initialRobotDirectionRange` | npm | `m.range(min = 0; max = 0)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |
| `targetXRange` | npm | `m.range(min = 0.5; max = 0.5)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |
| `targetYRange` | npm | `m.range(min = 0.15; max = 0.15)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |
| `robotRadius` | d | `0.05` | <code>double</code> |
| `robotMaxV` | d | `0.01` | <code>double</code> |
| `sensorsAngleRange` | npm | `m.range(min = -1.57; max = 1.57)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |
| `nOfSensors` | i | `5` | <code>int</code> |
| `sensorRange` | d | `0.5` | <code>double</code> |
| `senseTarget` | b | `true` | <code>boolean</code> |
| `arena` | e | `EMPTY` | <code><abbr title="io.github.ericmedvet.jsdynsym.control.navigation.Arena$Prepared">Arena$Prepared</abbr></code> |
| `rescaleInput` | b | `true` | <code>boolean</code> |
| `randomGenerator` | npm | `m.defaultRG()` | <code><abbr title="java.util.random.RandomGenerator">RandomGenerator</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jsdynsym.control.navigation.NavigationEnvironment">NavigationEnvironment</abbr></code>; built from `io.github.ericmedvet.jsdynsym.buildable.builders.Environments.navigation()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `dynamicalSystem.environment.pointNavigation()`

`ds.e.pointNavigation(name; initialRobotXRange; initialRobotYRange; targetXRange; targetYRange; robotMaxV; collisionBlock; arena; rescaleInput; randomGenerator)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `name` | s | interpolate `nav-{arena}` | <code><abbr title="java.lang.String">String</abbr></code> |
| `initialRobotXRange` | npm | `m.range(min = 0.45; max = 0.55)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |
| `initialRobotYRange` | npm | `m.range(min = 0.8; max = 0.85)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |
| `targetXRange` | npm | `m.range(min = 0.5; max = 0.5)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |
| `targetYRange` | npm | `m.range(min = 0.15; max = 0.15)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |
| `robotMaxV` | d | `0.01` | <code>double</code> |
| `collisionBlock` | d | `0.005` | <code>double</code> |
| `arena` | e | `EMPTY` | <code><abbr title="io.github.ericmedvet.jsdynsym.control.navigation.Arena$Prepared">Arena$Prepared</abbr></code> |
| `rescaleInput` | b | `true` | <code>boolean</code> |
| `randomGenerator` | npm | `m.defaultRG()` | <code><abbr title="java.util.random.RandomGenerator">RandomGenerator</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jsdynsym.control.navigation.PointNavigationEnvironment">PointNavigationEnvironment</abbr></code>; built from `io.github.ericmedvet.jsdynsym.buildable.builders.Environments.pointNavigation()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `dynamicalSystem.environment.pong()`

`ds.e.pong(name; racketsInitialYRange; racketsLength; racketsMaxDeltaPosition; ballInitialVelocity; ballMaxVelocity; ballInitialAngleRange; ballAccelerationRate; maxPercentageAngleAdjustment; arenaXLength; arenaYLength; precision; randomGenerator)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `name` | s | interpolate `pong` | <code><abbr title="java.lang.String">String</abbr></code> |
| `racketsInitialYRange` | npm | `m.range(min = 22; max = 28)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |
| `racketsLength` | d | `5.0` | <code>double</code> |
| `racketsMaxDeltaPosition` | d | `0.5` | <code>double</code> |
| `ballInitialVelocity` | d | `20.0` | <code>double</code> |
| `ballMaxVelocity` | d | `50.0` | <code>double</code> |
| `ballInitialAngleRange` | npm | `m.range(min = -0.4; max = 0.4)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |
| `ballAccelerationRate` | d | `1.1` | <code>double</code> |
| `maxPercentageAngleAdjustment` | d | `0.1` | <code>double</code> |
| `arenaXLength` | d | `60.0` | <code>double</code> |
| `arenaYLength` | d | `50.0` | <code>double</code> |
| `precision` | d | `1.0E-5` | <code>double</code> |
| `randomGenerator` | npm | `m.defaultRG()` | <code><abbr title="java.util.random.RandomGenerator">RandomGenerator</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jsdynsym.control.pong.PongEnvironment">PongEnvironment</abbr></code>; built from `io.github.ericmedvet.jsdynsym.buildable.builders.Environments.pong()` by jgea-experimenter:2.7.1-SNAPSHOT

## Package `dynamicalSystem.environment.navigation`

Aliases: `ds.e.n`, `ds.e.nav`, `ds.e.navigation`, `ds.env.n`, `ds.env.nav`, `ds.env.navigation`, `ds.environment.n`, `ds.environment.nav`, `ds.environment.navigation`, `dynSys.e.n`, `dynSys.e.nav`, `dynSys.e.navigation`, `dynSys.env.n`, `dynSys.env.nav`, `dynSys.env.navigation`, `dynSys.environment.n`, `dynSys.environment.nav`, `dynSys.environment.navigation`, `dynamicalSystem.e.n`, `dynamicalSystem.e.nav`, `dynamicalSystem.e.navigation`, `dynamicalSystem.env.n`, `dynamicalSystem.env.nav`, `dynamicalSystem.env.navigation`, `dynamicalSystem.environment.n`, `dynamicalSystem.environment.nav`, `dynamicalSystem.environment.navigation`

### Builder `dynamicalSystem.environment.navigation.arenaCoverage()`

`ds.e.n.arenaCoverage(of; xBins; yBins; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jsdynsym.control.Simulation$Outcome">Simulation$Outcome</abbr>&lt;<abbr title="io.github.ericmedvet.jsdynsym.control.SingleAgentTask$Step">SingleAgentTask$Step</abbr>&lt;double[], double[], <abbr title="io.github.ericmedvet.jsdynsym.control.navigation.State">State</abbr>&gt;&gt;&gt;</code> |
| `xBins` | i | `10` | <code>int</code> |
| `yBins` | i | `10` | <code>int</code> |
| `format` | s | `%5.3f` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, <abbr title="java.lang.Double">Double</abbr>&gt;</code>; built from `io.github.ericmedvet.jsdynsym.buildable.builders.NavigationFunctions.arenaCoverage()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `dynamicalSystem.environment.navigation.avgD()`

`ds.e.n.avgD(of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jsdynsym.control.Simulation$Outcome">Simulation$Outcome</abbr>&lt;<abbr title="io.github.ericmedvet.jsdynsym.control.SingleAgentTask$Step">SingleAgentTask$Step</abbr>&lt;double[], double[], <abbr title="io.github.ericmedvet.jsdynsym.control.navigation.State">State</abbr>&gt;&gt;&gt;</code> |
| `format` | s | `%5.3f` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, <abbr title="java.lang.Double">Double</abbr>&gt;</code>; built from `io.github.ericmedvet.jsdynsym.buildable.builders.NavigationFunctions.avgD()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `dynamicalSystem.environment.navigation.closestRobotP()`

`ds.e.n.closestRobotP(of; normalized)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jsdynsym.control.Simulation$Outcome">Simulation$Outcome</abbr>&lt;<abbr title="io.github.ericmedvet.jsdynsym.control.SingleAgentTask$Step">SingleAgentTask$Step</abbr>&lt;double[], double[], <abbr title="io.github.ericmedvet.jsdynsym.control.navigation.State">State</abbr>&gt;&gt;&gt;</code> |
| `normalized` | b | `true` | <code>boolean</code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.NamedFunction">NamedFunction</abbr>&lt;X, <abbr title="io.github.ericmedvet.jsdynsym.control.geometry.Point">Point</abbr>&gt;</code>; built from `io.github.ericmedvet.jsdynsym.buildable.builders.NavigationFunctions.closestRobotP()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `dynamicalSystem.environment.navigation.distanceFromTarget()`

`ds.e.n.distanceFromTarget(of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jsdynsym.control.navigation.State">State</abbr>&gt;</code> |
| `format` | s | `%5.3f` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, <abbr title="java.lang.Double">Double</abbr>&gt;</code>; built from `io.github.ericmedvet.jsdynsym.buildable.builders.NavigationFunctions.distanceFromTarget()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `dynamicalSystem.environment.navigation.finalD()`

`ds.e.n.finalD(of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jsdynsym.control.Simulation$Outcome">Simulation$Outcome</abbr>&lt;<abbr title="io.github.ericmedvet.jsdynsym.control.SingleAgentTask$Step">SingleAgentTask$Step</abbr>&lt;double[], double[], <abbr title="io.github.ericmedvet.jsdynsym.control.navigation.State">State</abbr>&gt;&gt;&gt;</code> |
| `format` | s | `%5.3f` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, <abbr title="java.lang.Double">Double</abbr>&gt;</code>; built from `io.github.ericmedvet.jsdynsym.buildable.builders.NavigationFunctions.finalD()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `dynamicalSystem.environment.navigation.finalRobotP()`

`ds.e.n.finalRobotP(of; normalized)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jsdynsym.control.Simulation$Outcome">Simulation$Outcome</abbr>&lt;<abbr title="io.github.ericmedvet.jsdynsym.control.SingleAgentTask$Step">SingleAgentTask$Step</abbr>&lt;double[], double[], <abbr title="io.github.ericmedvet.jsdynsym.control.navigation.State">State</abbr>&gt;&gt;&gt;</code> |
| `normalized` | b | `true` | <code>boolean</code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.NamedFunction">NamedFunction</abbr>&lt;X, <abbr title="io.github.ericmedvet.jsdynsym.control.geometry.Point">Point</abbr>&gt;</code>; built from `io.github.ericmedvet.jsdynsym.buildable.builders.NavigationFunctions.finalRobotP()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `dynamicalSystem.environment.navigation.finalTime()`

`ds.e.n.finalTime(of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jsdynsym.control.Simulation$Outcome">Simulation$Outcome</abbr>&lt;<abbr title="io.github.ericmedvet.jsdynsym.control.SingleAgentTask$Step">SingleAgentTask$Step</abbr>&lt;double[], double[], <abbr title="io.github.ericmedvet.jsdynsym.control.navigation.State">State</abbr>&gt;&gt;&gt;</code> |
| `format` | s | `%5.3f` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, <abbr title="java.lang.Double">Double</abbr>&gt;</code>; built from `io.github.ericmedvet.jsdynsym.buildable.builders.NavigationFunctions.finalTime()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `dynamicalSystem.environment.navigation.finalTimePlusD()`

`ds.e.n.finalTimePlusD(of; epsilon; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jsdynsym.control.Simulation$Outcome">Simulation$Outcome</abbr>&lt;<abbr title="io.github.ericmedvet.jsdynsym.control.SingleAgentTask$Step">SingleAgentTask$Step</abbr>&lt;double[], double[], <abbr title="io.github.ericmedvet.jsdynsym.control.navigation.State">State</abbr>&gt;&gt;&gt;</code> |
| `epsilon` | d | `0.01` | <code>double</code> |
| `format` | s | `%5.3f` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, <abbr title="java.lang.Double">Double</abbr>&gt;</code>; built from `io.github.ericmedvet.jsdynsym.buildable.builders.NavigationFunctions.finalTimePlusD()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `dynamicalSystem.environment.navigation.minD()`

`ds.e.n.minD(of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jsdynsym.control.Simulation$Outcome">Simulation$Outcome</abbr>&lt;<abbr title="io.github.ericmedvet.jsdynsym.control.SingleAgentTask$Step">SingleAgentTask$Step</abbr>&lt;double[], double[], <abbr title="io.github.ericmedvet.jsdynsym.control.navigation.State">State</abbr>&gt;&gt;&gt;</code> |
| `format` | s | `%5.3f` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, <abbr title="java.lang.Double">Double</abbr>&gt;</code>; built from `io.github.ericmedvet.jsdynsym.buildable.builders.NavigationFunctions.minD()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `dynamicalSystem.environment.navigation.x()`

`ds.e.n.x(of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jsdynsym.control.geometry.Point">Point</abbr>&gt;</code> |
| `format` | s | `%5.3f` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, <abbr title="java.lang.Double">Double</abbr>&gt;</code>; built from `io.github.ericmedvet.jsdynsym.buildable.builders.NavigationFunctions.x()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `dynamicalSystem.environment.navigation.y()`

`ds.e.n.y(of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jsdynsym.control.geometry.Point">Point</abbr>&gt;</code> |
| `format` | s | `%5.3f` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, <abbr title="java.lang.Double">Double</abbr>&gt;</code>; built from `io.github.ericmedvet.jsdynsym.buildable.builders.NavigationFunctions.y()` by jgea-experimenter:2.7.1-SNAPSHOT

## Package `dynamicalSystem.environment.pong`

Aliases: `ds.e.pong`, `ds.env.pong`, `ds.environment.pong`, `dynSys.e.pong`, `dynSys.env.pong`, `dynSys.environment.pong`, `dynamicalSystem.e.pong`, `dynamicalSystem.env.pong`, `dynamicalSystem.environment.pong`

### Builder `dynamicalSystem.environment.pong.numberOfCollisionsWithBall1()`

`ds.e.pong.numberOfCollisionsWithBall1(of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jsdynsym.control.Simulation$Outcome">Simulation$Outcome</abbr>&lt;<abbr title="io.github.ericmedvet.jsdynsym.control.HomogeneousBiAgentTask$Step">HomogeneousBiAgentTask$Step</abbr>&lt;double[], double[], <abbr title="io.github.ericmedvet.jsdynsym.control.pong.PongEnvironment$State">PongEnvironment$State</abbr>&gt;&gt;&gt;</code> |
| `format` | s | `%5.0f` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, <abbr title="java.lang.Double">Double</abbr>&gt;</code>; built from `io.github.ericmedvet.jsdynsym.buildable.builders.PongFunctions.numberOfCollisionsWithBall1()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `dynamicalSystem.environment.pong.numberOfCollisionsWithBall2()`

`ds.e.pong.numberOfCollisionsWithBall2(of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jsdynsym.control.Simulation$Outcome">Simulation$Outcome</abbr>&lt;<abbr title="io.github.ericmedvet.jsdynsym.control.HomogeneousBiAgentTask$Step">HomogeneousBiAgentTask$Step</abbr>&lt;double[], double[], <abbr title="io.github.ericmedvet.jsdynsym.control.pong.PongEnvironment$State">PongEnvironment$State</abbr>&gt;&gt;&gt;</code> |
| `format` | s | `%5.0f` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, <abbr title="java.lang.Double">Double</abbr>&gt;</code>; built from `io.github.ericmedvet.jsdynsym.buildable.builders.PongFunctions.numberOfCollisionsWithBall2()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `dynamicalSystem.environment.pong.score1()`

`ds.e.pong.score1(of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jsdynsym.control.Simulation$Outcome">Simulation$Outcome</abbr>&lt;<abbr title="io.github.ericmedvet.jsdynsym.control.HomogeneousBiAgentTask$Step">HomogeneousBiAgentTask$Step</abbr>&lt;double[], double[], <abbr title="io.github.ericmedvet.jsdynsym.control.pong.PongEnvironment$State">PongEnvironment$State</abbr>&gt;&gt;&gt;</code> |
| `format` | s | `%5.3f` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, <abbr title="java.lang.Double">Double</abbr>&gt;</code>; built from `io.github.ericmedvet.jsdynsym.buildable.builders.PongFunctions.score1()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `dynamicalSystem.environment.pong.score2()`

`ds.e.pong.score2(of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jsdynsym.control.Simulation$Outcome">Simulation$Outcome</abbr>&lt;<abbr title="io.github.ericmedvet.jsdynsym.control.HomogeneousBiAgentTask$Step">HomogeneousBiAgentTask$Step</abbr>&lt;double[], double[], <abbr title="io.github.ericmedvet.jsdynsym.control.pong.PongEnvironment$State">PongEnvironment$State</abbr>&gt;&gt;&gt;</code> |
| `format` | s | `%5.3f` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, <abbr title="java.lang.Double">Double</abbr>&gt;</code>; built from `io.github.ericmedvet.jsdynsym.buildable.builders.PongFunctions.score2()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `dynamicalSystem.environment.pong.scoreDiff1()`

`ds.e.pong.scoreDiff1(of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jsdynsym.control.Simulation$Outcome">Simulation$Outcome</abbr>&lt;<abbr title="io.github.ericmedvet.jsdynsym.control.HomogeneousBiAgentTask$Step">HomogeneousBiAgentTask$Step</abbr>&lt;double[], double[], <abbr title="io.github.ericmedvet.jsdynsym.control.pong.PongEnvironment$State">PongEnvironment$State</abbr>&gt;&gt;&gt;</code> |
| `format` | s | `%5.3f` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, <abbr title="java.lang.Double">Double</abbr>&gt;</code>; built from `io.github.ericmedvet.jsdynsym.buildable.builders.PongFunctions.scoreDiff1()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `dynamicalSystem.environment.pong.scoreDiff2()`

`ds.e.pong.scoreDiff2(of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jsdynsym.control.Simulation$Outcome">Simulation$Outcome</abbr>&lt;<abbr title="io.github.ericmedvet.jsdynsym.control.HomogeneousBiAgentTask$Step">HomogeneousBiAgentTask$Step</abbr>&lt;double[], double[], <abbr title="io.github.ericmedvet.jsdynsym.control.pong.PongEnvironment$State">PongEnvironment$State</abbr>&gt;&gt;&gt;</code> |
| `format` | s | `%5.3f` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, <abbr title="java.lang.Double">Double</abbr>&gt;</code>; built from `io.github.ericmedvet.jsdynsym.buildable.builders.PongFunctions.scoreDiff2()` by jgea-experimenter:2.7.1-SNAPSHOT

## Package `dynamicalSystem.function`

Aliases: `ds.f`, `ds.function`, `dynSys.f`, `dynSys.function`, `dynamicalSystem.f`, `dynamicalSystem.function`

### Builder `dynamicalSystem.function.doubleOp()`

`ds.f.doubleOp(of; activationF; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="java.lang.Double">Double</abbr>&gt;</code> |
| `activationF` | e | `IDENTITY` | <code><abbr title="io.github.ericmedvet.jsdynsym.core.numerical.ann.MultiLayerPerceptron$ActivationFunction">MultiLayerPerceptron$ActivationFunction</abbr></code> |
| `format` | s | `%.1f` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, <abbr title="java.lang.Double">Double</abbr>&gt;</code>; built from `io.github.ericmedvet.jsdynsym.buildable.builders.Functions.doubleOp()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `dynamicalSystem.function.opponentBiSimulator()`

`ds.f.opponentBiSimulator(of; simulation; opponent; home; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, S&gt;</code> |
| `simulation` | npm |  | <code><abbr title="io.github.ericmedvet.jsdynsym.control.HomogeneousBiSimulation">HomogeneousBiSimulation</abbr>&lt;S, SS, B&gt;</code> |
| `opponent` | npm |  | <code><abbr title="java.util.function.Supplier">Supplier</abbr>&lt;S&gt;</code> |
| `home` | b | `true` | <code>boolean</code> |
| `format` | s | `%s` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.NamedFunction">NamedFunction</abbr>&lt;X, <abbr title="io.github.ericmedvet.jsdynsym.control.Simulation$Outcome">Simulation$Outcome</abbr>&lt;SS&gt;&gt;</code>; built from `io.github.ericmedvet.jsdynsym.buildable.builders.Functions.opponentBiSimulator()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `dynamicalSystem.function.selfBiSimulator()`

`ds.f.selfBiSimulator(of; simulation; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, S&gt;</code> |
| `simulation` | npm |  | <code><abbr title="io.github.ericmedvet.jsdynsym.control.HomogeneousBiSimulation">HomogeneousBiSimulation</abbr>&lt;S, SS, B&gt;</code> |
| `format` | s | `%s` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.NamedFunction">NamedFunction</abbr>&lt;X, <abbr title="io.github.ericmedvet.jsdynsym.control.Simulation$Outcome">Simulation$Outcome</abbr>&lt;SS&gt;&gt;</code>; built from `io.github.ericmedvet.jsdynsym.buildable.builders.Functions.selfBiSimulator()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `dynamicalSystem.function.simOutcome()`

`ds.f.simOutcome(of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jsdynsym.control.Simulation$Outcome">Simulation$Outcome</abbr>&lt;S&gt;&gt;</code> |
| `format` | s | `%s` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.NamedFunction">NamedFunction</abbr>&lt;X, <abbr title="java.util.SortedMap">SortedMap</abbr>&lt;<abbr title="java.lang.Double">Double</abbr>, S&gt;&gt;</code>; built from `io.github.ericmedvet.jsdynsym.buildable.builders.Functions.simOutcome()` by jgea-experimenter:2.7.1-SNAPSHOT

## Package `dynamicalSystem.num`

Aliases: `ds.num`, `dynSys.num`, `dynamicalSystem.num`

### Builder `dynamicalSystem.num.drn()`

`ds.num.drn(timeRange; innerNeuronsRatio; activationFunction; threshold; timeResolution)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `timeRange` | npm | `m.range(min = 0; max = 1)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |
| `innerNeuronsRatio` | d | `1.0` | <code>double</code> |
| `activationFunction` | e | `TANH` | <code><abbr title="io.github.ericmedvet.jsdynsym.core.numerical.ann.MultiLayerPerceptron$ActivationFunction">MultiLayerPerceptron$ActivationFunction</abbr></code> |
| `threshold` | d | `0.1` | <code>double</code> |
| `timeResolution` | d | `0.16666` | <code>double</code> |

Produces <code><abbr title="io.github.ericmedvet.jsdynsym.buildable.builders.NumericalDynamicalSystems$Builder">NumericalDynamicalSystems$Builder</abbr>&lt;<abbr title="io.github.ericmedvet.jsdynsym.core.numerical.ann.DelayedRecurrentNetwork">DelayedRecurrentNetwork</abbr>, <abbr title="io.github.ericmedvet.jsdynsym.core.numerical.ann.DelayedRecurrentNetwork$State">DelayedRecurrentNetwork$State</abbr>&gt;</code>; built from `io.github.ericmedvet.jsdynsym.buildable.builders.NumericalDynamicalSystems.drn()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `dynamicalSystem.num.enhanced()`

`ds.num.enhanced(windowT; inner; types)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `windowT` | d |  | <code>double</code> |
| `inner` | npm |  | <code><abbr title="io.github.ericmedvet.jsdynsym.buildable.builders.NumericalDynamicalSystems$Builder">NumericalDynamicalSystems$Builder</abbr>&lt;? extends <abbr title="io.github.ericmedvet.jsdynsym.core.numerical.NumericalDynamicalSystem">NumericalDynamicalSystem</abbr>&lt;S&gt;, S&gt;</code> |
| `types` | e[] | `[CURRENT, TREND, AVG]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="io.github.ericmedvet.jsdynsym.core.numerical.EnhancedInput$Type">EnhancedInput$Type</abbr>&gt;</code> |

Produces <code><abbr title="io.github.ericmedvet.jsdynsym.buildable.builders.NumericalDynamicalSystems$Builder">NumericalDynamicalSystems$Builder</abbr>&lt;<abbr title="io.github.ericmedvet.jsdynsym.core.numerical.EnhancedInput">EnhancedInput</abbr>&lt;S&gt;, S&gt;</code>; built from `io.github.ericmedvet.jsdynsym.buildable.builders.NumericalDynamicalSystems.enhanced()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `dynamicalSystem.num.inStepped()`

`ds.num.inStepped(stepT; inner)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `stepT` | d | `1.0` | <code>double</code> |
| `inner` | npm |  | <code><abbr title="io.github.ericmedvet.jsdynsym.buildable.builders.NumericalDynamicalSystems$Builder">NumericalDynamicalSystems$Builder</abbr>&lt;? extends <abbr title="io.github.ericmedvet.jsdynsym.core.numerical.NumericalDynamicalSystem">NumericalDynamicalSystem</abbr>&lt;S&gt;, S&gt;</code> |

Produces <code><abbr title="io.github.ericmedvet.jsdynsym.buildable.builders.NumericalDynamicalSystems$Builder">NumericalDynamicalSystems$Builder</abbr>&lt;<abbr title="io.github.ericmedvet.jsdynsym.core.numerical.NumericalDynamicalSystem">NumericalDynamicalSystem</abbr>&lt;<abbr title="io.github.ericmedvet.jsdynsym.core.composed.Stepped$State">Stepped$State</abbr>&lt;S&gt;&gt;, <abbr title="io.github.ericmedvet.jsdynsym.core.composed.Stepped$State">Stepped$State</abbr>&lt;S&gt;&gt;</code>; built from `io.github.ericmedvet.jsdynsym.buildable.builders.NumericalDynamicalSystems.inStepped()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `dynamicalSystem.num.mlp()`

`ds.num.mlp(innerLayerRatio; nOfInnerLayers; activationFunction)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `innerLayerRatio` | d | `0.65` | <code>double</code> |
| `nOfInnerLayers` | i | `1` | <code>int</code> |
| `activationFunction` | e | `TANH` | <code><abbr title="io.github.ericmedvet.jsdynsym.core.numerical.ann.MultiLayerPerceptron$ActivationFunction">MultiLayerPerceptron$ActivationFunction</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jsdynsym.buildable.builders.NumericalDynamicalSystems$Builder">NumericalDynamicalSystems$Builder</abbr>&lt;<abbr title="io.github.ericmedvet.jsdynsym.core.numerical.ann.MultiLayerPerceptron">MultiLayerPerceptron</abbr>, <abbr title="io.github.ericmedvet.jsdynsym.core.StatelessSystem$State">StatelessSystem$State</abbr>&gt;</code>; built from `io.github.ericmedvet.jsdynsym.buildable.builders.NumericalDynamicalSystems.mlp()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `dynamicalSystem.num.noised()`

`ds.num.noised(inputSigma; outputSigma; randomGenerator; inner)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `inputSigma` | d | `0.01` | <code>double</code> |
| `outputSigma` | d | `0.01` | <code>double</code> |
| `randomGenerator` | npm | `m.defaultRG()` | <code><abbr title="java.util.random.RandomGenerator">RandomGenerator</abbr></code> |
| `inner` | npm |  | <code><abbr title="io.github.ericmedvet.jsdynsym.buildable.builders.NumericalDynamicalSystems$Builder">NumericalDynamicalSystems$Builder</abbr>&lt;? extends <abbr title="io.github.ericmedvet.jsdynsym.core.numerical.NumericalDynamicalSystem">NumericalDynamicalSystem</abbr>&lt;S&gt;, S&gt;</code> |

Produces <code><abbr title="io.github.ericmedvet.jsdynsym.buildable.builders.NumericalDynamicalSystems$Builder">NumericalDynamicalSystems$Builder</abbr>&lt;<abbr title="io.github.ericmedvet.jsdynsym.core.numerical.Noised">Noised</abbr>&lt;S&gt;, S&gt;</code>; built from `io.github.ericmedvet.jsdynsym.buildable.builders.NumericalDynamicalSystems.noised()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `dynamicalSystem.num.outStepped()`

`ds.num.outStepped(stepT; inner)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `stepT` | d | `1.0` | <code>double</code> |
| `inner` | npm |  | <code><abbr title="io.github.ericmedvet.jsdynsym.buildable.builders.NumericalDynamicalSystems$Builder">NumericalDynamicalSystems$Builder</abbr>&lt;? extends <abbr title="io.github.ericmedvet.jsdynsym.core.numerical.NumericalDynamicalSystem">NumericalDynamicalSystem</abbr>&lt;S&gt;, S&gt;</code> |

Produces <code><abbr title="io.github.ericmedvet.jsdynsym.buildable.builders.NumericalDynamicalSystems$Builder">NumericalDynamicalSystems$Builder</abbr>&lt;<abbr title="io.github.ericmedvet.jsdynsym.core.numerical.NumericalDynamicalSystem">NumericalDynamicalSystem</abbr>&lt;<abbr title="io.github.ericmedvet.jsdynsym.core.composed.Stepped$State">Stepped$State</abbr>&lt;S&gt;&gt;, <abbr title="io.github.ericmedvet.jsdynsym.core.composed.Stepped$State">Stepped$State</abbr>&lt;S&gt;&gt;</code>; built from `io.github.ericmedvet.jsdynsym.buildable.builders.NumericalDynamicalSystems.outStepped()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `dynamicalSystem.num.sin()`

`ds.num.sin(p; f; a; b)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `p` | npm | `m.range(min = -1.57; max = 1.57)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |
| `f` | npm | `m.range(min = 0; max = 1)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |
| `a` | npm | `m.range(min = 0; max = 1)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |
| `b` | npm | `m.range(min = -0.5; max = 0.5)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jsdynsym.buildable.builders.NumericalDynamicalSystems$Builder">NumericalDynamicalSystems$Builder</abbr>&lt;<abbr title="io.github.ericmedvet.jsdynsym.core.numerical.Sinusoidal">Sinusoidal</abbr>, <abbr title="io.github.ericmedvet.jsdynsym.core.StatelessSystem$State">StatelessSystem$State</abbr>&gt;</code>; built from `io.github.ericmedvet.jsdynsym.buildable.builders.NumericalDynamicalSystems.sin()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `dynamicalSystem.num.stepped()`

`ds.num.stepped(stepT; inner)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `stepT` | d | `0.1` | <code>double</code> |
| `inner` | npm |  | <code><abbr title="io.github.ericmedvet.jsdynsym.buildable.builders.NumericalDynamicalSystems$Builder">NumericalDynamicalSystems$Builder</abbr>&lt;? extends <abbr title="io.github.ericmedvet.jsdynsym.core.numerical.NumericalDynamicalSystem">NumericalDynamicalSystem</abbr>&lt;S&gt;, S&gt;</code> |

Produces <code><abbr title="io.github.ericmedvet.jsdynsym.buildable.builders.NumericalDynamicalSystems$Builder">NumericalDynamicalSystems$Builder</abbr>&lt;<abbr title="io.github.ericmedvet.jsdynsym.core.numerical.NumericalDynamicalSystem">NumericalDynamicalSystem</abbr>&lt;<abbr title="io.github.ericmedvet.jsdynsym.core.composed.Stepped$State">Stepped$State</abbr>&lt;S&gt;&gt;, <abbr title="io.github.ericmedvet.jsdynsym.core.composed.Stepped$State">Stepped$State</abbr>&lt;S&gt;&gt;</code>; built from `io.github.ericmedvet.jsdynsym.buildable.builders.NumericalDynamicalSystems.stepped()` by jgea-experimenter:2.7.1-SNAPSHOT

## Package `dynamicalSystem.opponent.pong`

Aliases: `ds.opponent.pong`, `dynSys.opponent.pong`, `dynamicalSystem.opponent.pong`

### Builder `dynamicalSystem.opponent.pong.simple()`

`ds.opponent.pong.simple(deltaPosition)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `deltaPosition` | d | `1.0` | <code>double</code> |

Produces <code><abbr title="java.util.function.Supplier">Supplier</abbr>&lt;<abbr title="io.github.ericmedvet.jsdynsym.control.pong.PongAgent">PongAgent</abbr>&gt;</code>; built from `io.github.ericmedvet.jsdynsym.buildable.builders.PongOpponents.simple()` by jgea-experimenter:2.7.1-SNAPSHOT

## Package `dynamicalSystem.simulation`

Aliases: `ds.s`, `ds.sim`, `ds.simulation`, `dynSys.s`, `dynSys.sim`, `dynSys.simulation`, `dynamicalSystem.s`, `dynamicalSystem.sim`, `dynamicalSystem.simulation`

### Builder `dynamicalSystem.simulation.variableSensorPositionsNavigation()`

`ds.s.variableSensorPositionsNavigation(name; initialRobotXRange; initialRobotYRange; initialRobotDirectionRange; targetXRange; targetYRange; robotRadius; robotMaxV; nOfSensors; sensorRange; senseTarget; arena; rescaleInput; sortAngles; randomGenerator; dT; initialT; finalT)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `name` | s | interpolate `vs[{nOfSensors}]-nav-{arena}` | <code><abbr title="java.lang.String">String</abbr></code> |
| `initialRobotXRange` | npm | `m.range(min = 0.45; max = 0.55)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |
| `initialRobotYRange` | npm | `m.range(min = 0.8; max = 0.85)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |
| `initialRobotDirectionRange` | npm | `m.range(min = 0; max = 0)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |
| `targetXRange` | npm | `m.range(min = 0.5; max = 0.5)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |
| `targetYRange` | npm | `m.range(min = 0.15; max = 0.15)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |
| `robotRadius` | d | `0.05` | <code>double</code> |
| `robotMaxV` | d | `0.01` | <code>double</code> |
| `nOfSensors` | i | `5` | <code>int</code> |
| `sensorRange` | d | `0.5` | <code>double</code> |
| `senseTarget` | b | `true` | <code>boolean</code> |
| `arena` | e | `EMPTY` | <code><abbr title="io.github.ericmedvet.jsdynsym.control.navigation.Arena$Prepared">Arena$Prepared</abbr></code> |
| `rescaleInput` | b | `true` | <code>boolean</code> |
| `sortAngles` | b | `true` | <code>boolean</code> |
| `randomGenerator` | npm | `m.defaultRG()` | <code><abbr title="java.util.random.RandomGenerator">RandomGenerator</abbr></code> |
| `dT` | d | `0.1` | <code>double</code> |
| `initialT` | d | `0.0` | <code>double</code> |
| `finalT` | d | `60.0` | <code>double</code> |

Produces <code><abbr title="io.github.ericmedvet.jsdynsym.control.navigation.VariableSensorPositionsNavigation">VariableSensorPositionsNavigation</abbr></code>; built from `io.github.ericmedvet.jsdynsym.buildable.builders.Simulations.variableSensorPositionsNavigation()` by jgea-experimenter:2.7.1-SNAPSHOT

## Package `dynamicalSystem.singleAgentTask`

Aliases: `ds.saTask`, `ds.sat`, `ds.singleAgentTask`, `dynSys.saTask`, `dynSys.sat`, `dynSys.singleAgentTask`, `dynamicalSystem.saTask`, `dynamicalSystem.sat`, `dynamicalSystem.singleAgentTask`

### Builder `dynamicalSystem.singleAgentTask.fromEnvironment()`

`ds.sat.fromEnvironment(name; environment; stopCondition; tRange; dT)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `name` | s | interpolate `{environment.name}[{tRange.min};{tRange.max}]` | <code><abbr title="java.lang.String">String</abbr></code> |
| `environment` | npm |  | <code><abbr title="io.github.ericmedvet.jsdynsym.control.Environment">Environment</abbr>&lt;O, A, S, C&gt;</code> |
| `stopCondition` | npm | `predicate.not(condition = predicate.always())` | <code><abbr title="java.util.function.Predicate">Predicate</abbr>&lt;S&gt;</code> |
| `tRange` | npm |  | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |
| `dT` | d |  | <code>double</code> |

Produces <code><abbr title="io.github.ericmedvet.jsdynsym.control.SingleAgentTask">SingleAgentTask</abbr>&lt;C, O, A, S&gt;</code>; built from `io.github.ericmedvet.jsdynsym.buildable.builders.SingleAgentTasks.fromEnvironment()` by jgea-experimenter:2.7.1-SNAPSHOT

## Package `ea`

### Builder `ea.experiment()`

`ea.experiment(name; startTime; runs; listeners)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `name` | s | `` | <code><abbr title="java.lang.String">String</abbr></code> |
| `startTime` | s | `` | <code><abbr title="java.lang.String">String</abbr></code> |
| `runs` | npm[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.experimenter.Run">Run</abbr>&lt;?, ?, ?, ?&gt;&gt;</code> |
| `listeners` | npm[] | `[ea.l.console()]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.util.function.BiFunction">BiFunction</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.experimenter.Experiment">Experiment</abbr>, <abbr title="java.util.concurrent.ExecutorService">ExecutorService</abbr>, <abbr title="io.github.ericmedvet.jgea.core.listener.ListenerFactory">ListenerFactory</abbr>&lt;? super <abbr title="io.github.ericmedvet.jgea.core.solver.POCPopulationState">POCPopulationState</abbr>&lt;?, ?, ?, ?, ?&gt;, <abbr title="io.github.ericmedvet.jgea.experimenter.Run">Run</abbr>&lt;?, ?, ?, ?&gt;&gt;&gt;&gt;</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.experimenter.Experiment">Experiment</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.Experiment()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.run()`

`ea.run(name; solver; problem; randomGenerator)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `name` | s | `` | <code><abbr title="java.lang.String">String</abbr></code> |
| `solver` | npm |  | <code><abbr title="java.util.function.Function">Function</abbr>&lt;S, ? extends <abbr title="io.github.ericmedvet.jgea.core.solver.AbstractPopulationBasedIterativeSolver">AbstractPopulationBasedIterativeSolver</abbr>&lt;? extends <abbr title="io.github.ericmedvet.jgea.core.solver.POCPopulationState">POCPopulationState</abbr>&lt;?, G, S, Q, P&gt;, P, ?, G, S, Q&gt;&gt;</code> |
| `problem` | npm |  | <code>P</code> |
| `randomGenerator` | npm |  | <code><abbr title="java.util.random.RandomGenerator">RandomGenerator</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.experimenter.Run">Run</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.Run()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.runOutcome()`

`ea.runOutcome(index; run; serializedGenotypes)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `index` | s |  | <code><abbr title="java.lang.String">String</abbr></code> |
| `run` | npm |  | <code><abbr title="io.github.ericmedvet.jgea.experimenter.Run">Run</abbr>&lt;?, ?, ?, ?&gt;</code> |
| `serializedGenotypes` | s[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.String">String</abbr>&gt;</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.experimenter.RunOutcome">RunOutcome</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.RunOutcome()` by jgea-experimenter:2.7.1-SNAPSHOT

## Package `ea.accumulator`

Aliases: `ea.a`, `ea.acc`, `ea.accumulator`

### Builder `ea.accumulator.all()`

`ea.a.all(eFunction; listFunction)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `eFunction` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;E, F&gt;</code> |
| `listFunction` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;<abbr title="java.util.List">List</abbr>&lt;F&gt;, O&gt;</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.core.listener.AccumulatorFactory">AccumulatorFactory</abbr>&lt;E, O, R&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Accumulators.all()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.accumulator.bests()`

`ea.a.bests(eFunction; listFunction)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `eFunction` | npm | `ea.f.best()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;E, F&gt;</code> |
| `listFunction` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;<abbr title="java.util.List">List</abbr>&lt;F&gt;, O&gt;</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.core.listener.AccumulatorFactory">AccumulatorFactory</abbr>&lt;E, O, R&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Accumulators.all()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.accumulator.first()`

`ea.a.first(eFunction; listFunction)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `eFunction` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;E, F&gt;</code> |
| `listFunction` | npm | `f.nTh(n = 1)` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;<abbr title="java.util.List">List</abbr>&lt;F&gt;, O&gt;</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.core.listener.AccumulatorFactory">AccumulatorFactory</abbr>&lt;E, O, R&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Accumulators.all()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.accumulator.last()`

`ea.a.last(eFunction; listFunction)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `eFunction` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;E, F&gt;</code> |
| `listFunction` | npm | `f.nTh(n = -1)` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;<abbr title="java.util.List">List</abbr>&lt;F&gt;, O&gt;</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.core.listener.AccumulatorFactory">AccumulatorFactory</abbr>&lt;E, O, R&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Accumulators.all()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.accumulator.lastBest()`

`ea.a.lastBest(eFunction; listFunction)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `eFunction` | npm | `ea.f.best()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;E, F&gt;</code> |
| `listFunction` | npm | `f.nTh(n = -1)` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;<abbr title="java.util.List">List</abbr>&lt;F&gt;, O&gt;</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.core.listener.AccumulatorFactory">AccumulatorFactory</abbr>&lt;E, O, R&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Accumulators.all()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.accumulator.lastPopulationMap()`

`ea.a.lastPopulationMap(serializerF)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `serializerF` | npm | `f.toBase64()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;<abbr title="java.lang.Object">Object</abbr>, <abbr title="java.lang.String">String</abbr>&gt;</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.core.listener.AccumulatorFactory">AccumulatorFactory</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.solver.POCPopulationState">POCPopulationState</abbr>&lt;?, G, ?, ?, ?&gt;, <abbr title="io.github.ericmedvet.jnb.core.NamedParamMap">NamedParamMap</abbr>, <abbr title="io.github.ericmedvet.jgea.experimenter.Run">Run</abbr>&lt;?, G, ?, ?&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Accumulators.lastPopulationMap()` by jgea-experimenter:2.7.1-SNAPSHOT

## Package `ea.biproblem.synthetic`

Aliases: `ea.biproblem.s`, `ea.biproblem.synthetic`, `ea.bp.s`, `ea.bp.synthetic`

### Builder `ea.biproblem.synthetic.pointAimingBiProblem()`

`ea.bp.s.pointAimingBiProblem(name; p; target)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `name` | s | interpolate `pointAimingBiProblem-{p}-{target}` | <code><abbr title="java.lang.String">String</abbr></code> |
| `p` | i | `100` | <code>int</code> |
| `target` | d | `1.0` | <code>double</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.problem.synthetic.numerical.PointAimingBiProblem">PointAimingBiProblem</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.SyntheticBiProblems.pointAimingBiProblem()` by jgea-experimenter:2.7.1-SNAPSHOT

## Package `ea.comparator`

### Builder `ea.comparator.ascending()`

`ea.comparator.ascending(of)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super X, ? extends T&gt;</code> |

Produces <code><abbr title="java.util.Comparator">Comparator</abbr>&lt;X&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Comparators.ascending()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.comparator.descending()`

`ea.comparator.descending(of)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super X, ? extends T&gt;</code> |

Produces <code><abbr title="java.util.Comparator">Comparator</abbr>&lt;X&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Comparators.descending()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.comparator.pAscending()`

`ea.comparator.pAscending(of)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super X, ? extends T&gt;</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.core.order.PartialComparator">PartialComparator</abbr>&lt;X&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Comparators.pAscending()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.comparator.pDescending()`

`ea.comparator.pDescending(of)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super X, ? extends T&gt;</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.core.order.PartialComparator">PartialComparator</abbr>&lt;X&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Comparators.pDescending()` by jgea-experimenter:2.7.1-SNAPSHOT

## Package `ea.consumer`

Aliases: `ea.c`, `ea.consumer`

### Builder `ea.consumer.composed()`

`ea.c.composed(of; f; c)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, Y&gt;</code> |
| `f` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;Y, O&gt;</code> |
| `c` | npm |  | <code><abbr title="io.github.ericmedvet.jnb.datastructure.TriConsumer">TriConsumer</abbr>&lt;O, <abbr title="io.github.ericmedvet.jgea.experimenter.Run">Run</abbr>&lt;?, ?, ?, ?&gt;, <abbr title="io.github.ericmedvet.jgea.experimenter.Experiment">Experiment</abbr>&gt;</code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.TriConsumer">TriConsumer</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.experimenter.Run">Run</abbr>&lt;?, ?, ?, ?&gt;, <abbr title="io.github.ericmedvet.jgea.experimenter.Experiment">Experiment</abbr>&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Consumers.composed()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.consumer.deaf()`

`ea.c.deaf()`

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.TriConsumer">TriConsumer</abbr>&lt;?, ?, ?&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Consumers.deaf()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.consumer.saver()`

`ea.c.saver(of; overwrite; path)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, O&gt;</code> |
| `overwrite` | b | `false` | <code>boolean</code> |
| `path` | s | `run-{run.index:%04d}` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.TriConsumer">TriConsumer</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.experimenter.Run">Run</abbr>&lt;?, ?, ?, ?&gt;, <abbr title="io.github.ericmedvet.jgea.experimenter.Experiment">Experiment</abbr>&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Consumers.saver()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.consumer.telegram()`

`ea.c.telegram(of; title; chatId; botIdFilePath)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, O&gt;</code> |
| `title` | s | `Experiment:
	{name}
Run {run.index}:
	Solver: {run.solver.name}
	Problem: {run.problem.name}
	Seed: {run.randomGenerator.seed}` | <code><abbr title="java.lang.String">String</abbr></code> |
| `chatId` | s |  | <code><abbr title="java.lang.String">String</abbr></code> |
| `botIdFilePath` | s |  | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.TriConsumer">TriConsumer</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.experimenter.Run">Run</abbr>&lt;?, ?, ?, ?&gt;, <abbr title="io.github.ericmedvet.jgea.experimenter.Experiment">Experiment</abbr>&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Consumers.telegram()` by jgea-experimenter:2.7.1-SNAPSHOT

## Package `ea.drawer`

Aliases: `ea.d`, `ea.drawer`

### Builder `ea.drawer.polyomino()`

`ea.d.polyomino(maxW; maxH; colors; borderColor)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `maxW` | i | `0` | <code>int</code> |
| `maxH` | i | `0` | <code>int</code> |
| `colors` | npm | `ea.misc.map(entries = [])` | <code><abbr title="java.util.Map">Map</abbr>&lt;<abbr title="java.lang.Character">Character</abbr>, <abbr title="java.awt.Color">Color</abbr>&gt;</code> |
| `borderColor` | npm | `ea.misc.colorByName(name = white)` | <code><abbr title="java.awt.Color">Color</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.experimenter.drawer.PolyominoDrawer">PolyominoDrawer</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Drawers.polyomino()` by jgea-experimenter:2.7.1-SNAPSHOT

## Package `ea.function`

Aliases: `ea.f`, `ea.function`

### Builder `ea.function.all()`

`ea.f.all(of)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.solver.POCPopulationState">POCPopulationState</abbr>&lt;I, G, S, Q, ?&gt;&gt;</code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.NamedFunction">NamedFunction</abbr>&lt;X, <abbr title="java.util.Collection">Collection</abbr>&lt;I&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Functions.all()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.function.archiveCoverage()`

`ea.f.archiveCoverage(of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.solver.mapelites.Archive">Archive</abbr>&lt;G&gt;&gt;</code> |
| `format` | s | `%4.2f` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, <abbr title="java.lang.Double">Double</abbr>&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Functions.archiveCoverage()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.function.archiveToGrid()`

`ea.f.archiveToGrid(of)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.solver.mapelites.Archive">Archive</abbr>&lt;G&gt;&gt;</code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.NamedFunction">NamedFunction</abbr>&lt;X, <abbr title="io.github.ericmedvet.jnb.datastructure.Grid">Grid</abbr>&lt;G&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Functions.archiveToGrid()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.function.behavior()`

`ea.f.behavior(of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.problem.BehaviorBasedProblem$Outcome">BehaviorBasedProblem$Outcome</abbr>&lt;B, ?&gt;&gt;</code> |
| `format` | s | `%s` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, B&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Functions.behavior()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.function.behaviorQuality()`

`ea.f.behaviorQuality(of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.problem.BehaviorBasedProblem$Outcome">BehaviorBasedProblem$Outcome</abbr>&lt;?, BQ&gt;&gt;</code> |
| `format` | s | `%s` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, BQ&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Functions.behaviorQuality()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.function.best()`

`ea.f.best(of)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.solver.POCPopulationState">POCPopulationState</abbr>&lt;I, G, S, Q, ?&gt;&gt;</code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.NamedFunction">NamedFunction</abbr>&lt;X, I&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Functions.best()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.function.coMeArchive1()`

`ea.f.coMeArchive1(of)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.solver.mapelites.CoMEPopulationState">CoMEPopulationState</abbr>&lt;G, ?, S, ?, ?, Q, ?&gt;&gt;</code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.NamedFunction">NamedFunction</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.solver.mapelites.Archive">Archive</abbr>&lt;? extends <abbr title="io.github.ericmedvet.jgea.core.solver.mapelites.MEIndividual">MEIndividual</abbr>&lt;G, S, Q&gt;&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Functions.coMeArchive1()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.function.coMeArchive2()`

`ea.f.coMeArchive2(of)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.solver.mapelites.CoMEPopulationState">CoMEPopulationState</abbr>&lt;?, G, ?, S, ?, Q, ?&gt;&gt;</code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.NamedFunction">NamedFunction</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.solver.mapelites.Archive">Archive</abbr>&lt;? extends <abbr title="io.github.ericmedvet.jgea.core.solver.mapelites.MEIndividual">MEIndividual</abbr>&lt;G, S, Q&gt;&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Functions.coMeArchive2()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.function.coMeStrategy1Field()`

`ea.f.coMeStrategy1Field(of; relative)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.solver.mapelites.CoMEPopulationState">CoMEPopulationState</abbr>&lt;?, ?, ?, ?, ?, ?, ?&gt;&gt;</code> |
| `relative` | b | `true` | <code>boolean</code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.NamedFunction">NamedFunction</abbr>&lt;X, <abbr title="java.util.Map">Map</abbr>&lt;<abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.Double">Double</abbr>&gt;, <abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.Double">Double</abbr>&gt;&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Functions.coMeStrategy1Field()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.function.coMeStrategy2Field()`

`ea.f.coMeStrategy2Field(of; relative)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.solver.mapelites.CoMEPopulationState">CoMEPopulationState</abbr>&lt;?, ?, ?, ?, ?, ?, ?&gt;&gt;</code> |
| `relative` | b | `true` | <code>boolean</code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.NamedFunction">NamedFunction</abbr>&lt;X, <abbr title="java.util.Map">Map</abbr>&lt;<abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.Double">Double</abbr>&gt;, <abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.Double">Double</abbr>&gt;&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Functions.coMeStrategy2Field()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.function.csvPlotter()`

`ea.f.csvPlotter(of; columnNameJoiner; doubleFormat; delimiter; missingDataString; mode)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, P&gt;</code> |
| `columnNameJoiner` | s | `.` | <code><abbr title="java.lang.String">String</abbr></code> |
| `doubleFormat` | s | `%.3e` | <code><abbr title="java.lang.String">String</abbr></code> |
| `delimiter` | s | `	` | <code><abbr title="java.lang.String">String</abbr></code> |
| `missingDataString` | s | `nan` | <code><abbr title="java.lang.String">String</abbr></code> |
| `mode` | e | `PAPER_FRIENDLY` | <code><abbr title="io.github.ericmedvet.jviz.core.plot.csv.Configuration$Mode">Configuration$Mode</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.NamedFunction">NamedFunction</abbr>&lt;X, <abbr title="java.lang.String">String</abbr>&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Functions.csvPlotter()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.function.descBin()`

`ea.f.descBin(descriptor; of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `descriptor` | npm |  | <code><abbr title="io.github.ericmedvet.jgea.core.solver.mapelites.MapElites$Descriptor">MapElites$Descriptor</abbr>&lt;G, S, Q&gt;</code> |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.solver.Individual">Individual</abbr>&lt;G, S, Q&gt;&gt;</code> |
| `format` | s | `%2d` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, <abbr title="java.lang.Integer">Integer</abbr>&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Functions.descBin()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.function.elapsedSecs()`

`ea.f.elapsedSecs(of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.solver.State">State</abbr>&lt;?, ?&gt;&gt;</code> |
| `format` | s | `%6.1f` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, <abbr title="java.lang.Double">Double</abbr>&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Functions.elapsedSecs()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.function.firsts()`

`ea.f.firsts(of)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.solver.POCPopulationState">POCPopulationState</abbr>&lt;I, G, S, Q, ?&gt;&gt;</code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.NamedFunction">NamedFunction</abbr>&lt;X, <abbr title="java.util.Collection">Collection</abbr>&lt;I&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Functions.firsts()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.function.fromProblem()`

`ea.f.fromProblem(of; problem; name; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, S&gt;</code> |
| `problem` | npm |  | <code><abbr title="io.github.ericmedvet.jgea.core.problem.QualityBasedProblem">QualityBasedProblem</abbr>&lt;S, Q&gt;</code> |
| `name` | s | interpolate `{problem.name}` | <code><abbr title="java.lang.String">String</abbr></code> |
| `format` | s | `%s` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, Q&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Functions.fromProblem()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.function.genotype()`

`ea.f.genotype(of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.solver.Individual">Individual</abbr>&lt;G, ?, ?&gt;&gt;</code> |
| `format` | s | `%s` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, G&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Functions.genotype()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.function.hist()`

`ea.f.hist(nOfBins; of)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `nOfBins` | i | `8` | <code>int</code> |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="java.util.Collection">Collection</abbr>&lt;<abbr title="java.lang.Number">Number</abbr>&gt;&gt;</code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.util.TextPlotter$Miniplot">TextPlotter$Miniplot</abbr>&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Functions.hist()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.function.hypervolume2D()`

`ea.f.hypervolume2D(minReference; maxReference; of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `minReference` | d[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.Double">Double</abbr>&gt;</code> |
| `maxReference` | d[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.Double">Double</abbr>&gt;</code> |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="java.util.Collection">Collection</abbr>&lt;<abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.Double">Double</abbr>&gt;&gt;&gt;</code> |
| `format` | s | `%.2f` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, <abbr title="java.lang.Double">Double</abbr>&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Functions.hypervolume2D()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.function.id()`

`ea.f.id(of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.solver.Individual">Individual</abbr>&lt;?, ?, ?&gt;&gt;</code> |
| `format` | s | `%6d` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, <abbr title="java.lang.Long">Long</abbr>&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Functions.id()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.function.imagePlotter()`

`ea.f.imagePlotter(of; w; h; axesShow; titlesShow; independences; freeScales; secondary)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, P&gt;</code> |
| `w` | i | `-1` | <code>int</code> |
| `h` | i | `-1` | <code>int</code> |
| `axesShow` | e | `BORDER` | <code><abbr title="io.github.ericmedvet.jviz.core.plot.image.Configuration$PlotMatrix$Show">Configuration$PlotMatrix$Show</abbr></code> |
| `titlesShow` | e | `BORDER` | <code><abbr title="io.github.ericmedvet.jviz.core.plot.image.Configuration$PlotMatrix$Show">Configuration$PlotMatrix$Show</abbr></code> |
| `independences` | e[] | `[ROWS, COLS]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="io.github.ericmedvet.jviz.core.plot.image.Configuration$PlotMatrix$Independence">Configuration$PlotMatrix$Independence</abbr>&gt;</code> |
| `freeScales` | b | `false` | <code>boolean</code> |
| `secondary` | b | `false` | <code>boolean</code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.NamedFunction">NamedFunction</abbr>&lt;X, <abbr title="java.awt.image.BufferedImage">BufferedImage</abbr>&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Functions.imagePlotter()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.function.lasts()`

`ea.f.lasts(of)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.solver.POCPopulationState">POCPopulationState</abbr>&lt;I, G, S, Q, ?&gt;&gt;</code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.NamedFunction">NamedFunction</abbr>&lt;X, <abbr title="java.util.Collection">Collection</abbr>&lt;I&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Functions.lasts()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.function.maMeArchive()`

`ea.f.maMeArchive(of; n)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.solver.mapelites.MAMEPopulationState">MAMEPopulationState</abbr>&lt;G, S, Q, ?&gt;&gt;</code> |
| `n` | i |  | <code>int</code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.NamedFunction">NamedFunction</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.solver.mapelites.Archive">Archive</abbr>&lt;? extends <abbr title="io.github.ericmedvet.jgea.core.solver.mapelites.MEIndividual">MEIndividual</abbr>&lt;G, S, Q&gt;&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Functions.maMeArchive()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.function.meArchive()`

`ea.f.meArchive(of)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.solver.mapelites.MEPopulationState">MEPopulationState</abbr>&lt;G, S, Q, ?&gt;&gt;</code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.NamedFunction">NamedFunction</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.solver.mapelites.Archive">Archive</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.solver.mapelites.MEIndividual">MEIndividual</abbr>&lt;G, S, Q&gt;&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Functions.meArchive()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.function.meBin()`

`ea.f.meBin(of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.solver.mapelites.MapElites$Descriptor$Coordinate">MapElites$Descriptor$Coordinate</abbr>&gt;</code> |
| `format` | s | `%3d` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, <abbr title="java.lang.Integer">Integer</abbr>&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Functions.meBin()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.function.meCoordinates()`

`ea.f.meCoordinates(of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.solver.mapelites.MEIndividual">MEIndividual</abbr>&lt;?, ?, ?&gt;&gt;</code> |
| `format` | s | `%s` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, <abbr title="java.util.List">List</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.solver.mapelites.MapElites$Descriptor$Coordinate">MapElites$Descriptor$Coordinate</abbr>&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Functions.meCoordinates()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.function.meValue()`

`ea.f.meValue(of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.solver.mapelites.MapElites$Descriptor$Coordinate">MapElites$Descriptor$Coordinate</abbr>&gt;</code> |
| `format` | s | `%.2f` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, <abbr title="java.lang.Double">Double</abbr>&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Functions.meValue()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.function.mids()`

`ea.f.mids(of)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.solver.POCPopulationState">POCPopulationState</abbr>&lt;I, G, S, Q, ?&gt;&gt;</code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.NamedFunction">NamedFunction</abbr>&lt;X, <abbr title="java.util.Collection">Collection</abbr>&lt;I&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Functions.mids()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.function.nOfBirths()`

`ea.f.nOfBirths(of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.solver.POCPopulationState">POCPopulationState</abbr>&lt;?, ?, ?, ?, ?&gt;&gt;</code> |
| `format` | s | `%5d` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, <abbr title="java.lang.Long">Long</abbr>&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Functions.nOfBirths()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.function.nOfEvals()`

`ea.f.nOfEvals(of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.solver.POCPopulationState">POCPopulationState</abbr>&lt;?, ?, ?, ?, ?&gt;&gt;</code> |
| `format` | s | `%5d` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, <abbr title="java.lang.Long">Long</abbr>&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Functions.nOfEvals()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.function.nOfIterations()`

`ea.f.nOfIterations(of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.solver.State">State</abbr>&lt;?, ?&gt;&gt;</code> |
| `format` | s | `%4d` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, <abbr title="java.lang.Long">Long</abbr>&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Functions.nOfIterations()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.function.overallTargetDistance()`

`ea.f.overallTargetDistance(of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.solver.POCPopulationState">POCPopulationState</abbr>&lt;?, ?, S, ?, P&gt;&gt;</code> |
| `format` | s | `%.2f` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, <abbr title="java.lang.Double">Double</abbr>&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Functions.overallTargetDistance()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.function.parentIds()`

`ea.f.parentIds(of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.solver.Individual">Individual</abbr>&lt;?, ?, ?&gt;&gt;</code> |
| `format` | s | `%s` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, <abbr title="java.util.Collection">Collection</abbr>&lt;<abbr title="java.lang.Long">Long</abbr>&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Functions.parentIds()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.function.popTargetDistances()`

`ea.f.popTargetDistances(of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.solver.POCPopulationState">POCPopulationState</abbr>&lt;?, ?, S, ?, P&gt;&gt;</code> |
| `format` | s | `%.2f` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, <abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.Double">Double</abbr>&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Functions.popTargetDistances()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.function.problem()`

`ea.f.problem(of)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.solver.State">State</abbr>&lt;P, S&gt;&gt;</code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.NamedFunction">NamedFunction</abbr>&lt;X, P&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Functions.problem()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.function.progress()`

`ea.f.progress(of)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.solver.State">State</abbr>&lt;?, ?&gt;&gt;</code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.NamedFunction">NamedFunction</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.util.Progress">Progress</abbr>&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Functions.progress()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.function.quality()`

`ea.f.quality(of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.solver.Individual">Individual</abbr>&lt;?, ?, Q&gt;&gt;</code> |
| `format` | s | `%s` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, Q&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Functions.quality()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.function.rate()`

`ea.f.rate(of)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.util.Progress">Progress</abbr>&gt;</code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.NamedFunction">NamedFunction</abbr>&lt;X, <abbr title="java.lang.Double">Double</abbr>&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Functions.rate()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.function.runKey()`

`ea.f.runKey(name; key; of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `name` | s | interpolate `{key}` | <code><abbr title="java.lang.String">String</abbr></code> |
| `key` | s |  | <code><abbr title="java.lang.String">String</abbr></code> |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.experimenter.Run">Run</abbr>&lt;?, ?, ?, ?&gt;&gt;</code> |
| `format` | s | `%s` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, <abbr title="java.lang.String">String</abbr>&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Functions.runKey()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.function.runString()`

`ea.f.runString(name; s; of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `name` | s | interpolate `{s}` | <code><abbr title="java.lang.String">String</abbr></code> |
| `s` | s |  | <code><abbr title="java.lang.String">String</abbr></code> |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.experimenter.Run">Run</abbr>&lt;?, ?, ?, ?&gt;&gt;</code> |
| `format` | s | `%s` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, <abbr title="java.lang.String">String</abbr>&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Functions.runString()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.function.size()`

`ea.f.size(of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="java.lang.Object">Object</abbr>&gt;</code> |
| `format` | s | `%d` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, <abbr title="java.lang.Integer">Integer</abbr>&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Functions.size()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.function.solution()`

`ea.f.solution(of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.solver.Individual">Individual</abbr>&lt;?, S, ?&gt;&gt;</code> |
| `format` | s | `%s` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, S&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Functions.solution()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.function.stateGrid()`

`ea.f.stateGrid(of)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.solver.cabea.GridPopulationState">GridPopulationState</abbr>&lt;G, S, Q, ?&gt;&gt;</code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.NamedFunction">NamedFunction</abbr>&lt;X, <abbr title="io.github.ericmedvet.jnb.datastructure.Grid">Grid</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.solver.Individual">Individual</abbr>&lt;G, S, Q&gt;&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Functions.stateGrid()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.function.supplied()`

`ea.f.supplied(of)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="java.util.function.Supplier">Supplier</abbr>&lt;Z&gt;&gt;</code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.NamedFunction">NamedFunction</abbr>&lt;X, Z&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Functions.supplied()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.function.targetDistances()`

`ea.f.targetDistances(problem; of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `problem` | npm |  | <code>P</code> |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.solver.Individual">Individual</abbr>&lt;?, S, ?&gt;&gt;</code> |
| `format` | s | `%.2f` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, <abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.Double">Double</abbr>&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Functions.targetDistances()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.function.toDoubleString()`

`ea.f.toDoubleString(of)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, Z&gt;</code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.NamedFunction">NamedFunction</abbr>&lt;X, <abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.Double">Double</abbr>&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Functions.toDoubleString()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.function.toImage()`

`ea.f.toImage(of; image; w; h)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, D&gt;</code> |
| `image` | npm |  | <code><abbr title="io.github.ericmedvet.jviz.core.drawer.ImageBuilder">ImageBuilder</abbr>&lt;D&gt;</code> |
| `w` | i | `-1` | <code>int</code> |
| `h` | i | `-1` | <code>int</code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.NamedFunction">NamedFunction</abbr>&lt;X, <abbr title="java.awt.image.BufferedImage">BufferedImage</abbr>&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Functions.toImage()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.function.toImagesVideo()`

`ea.f.toImagesVideo(of; image; w; h; frameRate; encoder)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="java.util.List">List</abbr>&lt;D&gt;&gt;</code> |
| `image` | npm |  | <code><abbr title="io.github.ericmedvet.jviz.core.drawer.ImageBuilder">ImageBuilder</abbr>&lt;D&gt;</code> |
| `w` | i | `-1` | <code>int</code> |
| `h` | i | `-1` | <code>int</code> |
| `frameRate` | d | `10.0` | <code>double</code> |
| `encoder` | e | `DEFAULT` | <code><abbr title="io.github.ericmedvet.jviz.core.util.VideoUtils$EncoderFacility">VideoUtils$EncoderFacility</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.NamedFunction">NamedFunction</abbr>&lt;X, <abbr title="io.github.ericmedvet.jviz.core.drawer.Video">Video</abbr>&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Functions.toImagesVideo()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.function.toVideo()`

`ea.f.toVideo(of; video; w; h; encoder)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, D&gt;</code> |
| `video` | npm |  | <code><abbr title="io.github.ericmedvet.jviz.core.drawer.VideoBuilder">VideoBuilder</abbr>&lt;D&gt;</code> |
| `w` | i | `-1` | <code>int</code> |
| `h` | i | `-1` | <code>int</code> |
| `encoder` | e | `DEFAULT` | <code><abbr title="io.github.ericmedvet.jviz.core.util.VideoUtils$EncoderFacility">VideoUtils$EncoderFacility</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.NamedFunction">NamedFunction</abbr>&lt;X, <abbr title="io.github.ericmedvet.jviz.core.drawer.Video">Video</abbr>&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Functions.toVideo()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.function.treeDepth()`

`ea.f.treeDepth(of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.representation.tree.Tree">Tree</abbr>&lt;C&gt;&gt;</code> |
| `format` | s | `%3d` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, <abbr title="java.lang.Integer">Integer</abbr>&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Functions.treeDepth()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.function.treeLabels()`

`ea.f.treeLabels(of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.representation.tree.Tree">Tree</abbr>&lt;C&gt;&gt;</code> |
| `format` | s | `%s` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, <abbr title="java.util.Collection">Collection</abbr>&lt;C&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Functions.treeLabels()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.function.treeLeaves()`

`ea.f.treeLeaves(of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.representation.tree.Tree">Tree</abbr>&lt;C&gt;&gt;</code> |
| `format` | s | `%s` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, <abbr title="java.util.Collection">Collection</abbr>&lt;C&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Functions.treeLeaves()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.function.treeSize()`

`ea.f.treeSize(of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.representation.tree.Tree">Tree</abbr>&lt;C&gt;&gt;</code> |
| `format` | s | `%3d` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, <abbr title="java.lang.Integer">Integer</abbr>&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Functions.treeSize()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.function.ttpnDeadGatesRate()`

`ea.f.ttpnDeadGatesRate(of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.representation.programsynthesis.ttpn.Network">Network</abbr>&gt;</code> |
| `format` | s | `%5.3f` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, <abbr title="java.lang.Double">Double</abbr>&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Functions.ttpnDeadGatesRate()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.function.ttpnDeadOrIUnwiredOutputGatesRate()`

`ea.f.ttpnDeadOrIUnwiredOutputGatesRate(of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.representation.programsynthesis.ttpn.Network">Network</abbr>&gt;</code> |
| `format` | s | `%5.3f` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, <abbr title="java.lang.Double">Double</abbr>&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Functions.ttpnDeadOrIUnwiredOutputGatesRate()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.function.ttpnNOfTypes()`

`ea.f.ttpnNOfTypes(of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.representation.programsynthesis.ttpn.Network">Network</abbr>&gt;</code> |
| `format` | s | `%5.3f` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, <abbr title="java.lang.Integer">Integer</abbr>&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Functions.ttpnNOfTypes()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.function.validationQuality()`

`ea.f.validationQuality(of; individual; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.solver.POCPopulationState">POCPopulationState</abbr>&lt;?, ?, S, Q, P&gt;&gt;</code> |
| `individual` | npm | `ea.f.best()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.solver.POCPopulationState">POCPopulationState</abbr>&lt;?, ?, S, Q, P&gt;, <abbr title="io.github.ericmedvet.jgea.core.solver.Individual">Individual</abbr>&lt;?, S, Q&gt;&gt;</code> |
| `format` | s | `%s` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, Q&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Functions.validationQuality()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.function.videoPlotter()`

`ea.f.videoPlotter(of; w; h; encoder; frameRate; freeScales; secondary)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, P&gt;</code> |
| `w` | i | `-1` | <code>int</code> |
| `h` | i | `-1` | <code>int</code> |
| `encoder` | e | `DEFAULT` | <code><abbr title="io.github.ericmedvet.jviz.core.util.VideoUtils$EncoderFacility">VideoUtils$EncoderFacility</abbr></code> |
| `frameRate` | d | `10.0` | <code>double</code> |
| `freeScales` | b | `false` | <code>boolean</code> |
| `secondary` | b | `false` | <code>boolean</code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.NamedFunction">NamedFunction</abbr>&lt;X, <abbr title="io.github.ericmedvet.jviz.core.drawer.Video">Video</abbr>&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Functions.videoPlotter()` by jgea-experimenter:2.7.1-SNAPSHOT

## Package `ea.grammar`

### Builder `ea.grammar.fromProblem()`

`ea.grammar.fromProblem(problem)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `problem` | npm |  | <code><abbr title="io.github.ericmedvet.jgea.core.representation.grammar.string.GrammarBasedProblem">GrammarBasedProblem</abbr>&lt;N, ?&gt;</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.core.representation.grammar.string.StringGrammar">StringGrammar</abbr>&lt;N&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Grammars.fromProblem()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.grammar.gridBundled()`

`ea.grammar.gridBundled(name)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `name` | s |  | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.core.representation.grammar.grid.GridGrammar">GridGrammar</abbr>&lt;<abbr title="java.lang.Character">Character</abbr>&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Grammars.gridBundled()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.grammar.gridFile()`

`ea.grammar.gridFile(path)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `path` | s |  | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.core.representation.grammar.grid.GridGrammar">GridGrammar</abbr>&lt;<abbr title="java.lang.String">String</abbr>&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Grammars.gridFile()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.grammar.regression()`

`ea.grammar.regression(constants; operators; problem)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `constants` | d[] | `[0.1, 1.0, 10.0]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.Double">Double</abbr>&gt;</code> |
| `operators` | e[] | `[+, -, *, p÷, plog]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.representation.tree.numeric.Element$Operator">Element$Operator</abbr>&gt;</code> |
| `problem` | npm |  | <code><abbr title="io.github.ericmedvet.jgea.core.problem.Problem">Problem</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.representation.NamedUnivariateRealFunction">NamedUnivariateRealFunction</abbr>&gt;</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.core.representation.grammar.string.StringGrammar">StringGrammar</abbr>&lt;<abbr title="java.lang.String">String</abbr>&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Grammars.regression()` by jgea-experimenter:2.7.1-SNAPSHOT

## Package `ea.listener`

Aliases: `ea.l`, `ea.listener`

### Builder `ea.listener.allCsv()`

`ea.l.allCsv(path; errorString; intFormat; doubleFormat; defaultFunctions; functions; individualFunctions; defaultRunFunctions; runFunctions; deferred; onlyLast; condition)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `path` | s |  | <code><abbr title="java.lang.String">String</abbr></code> |
| `errorString` | s | `NA` | <code><abbr title="java.lang.String">String</abbr></code> |
| `intFormat` | s | `%d` | <code><abbr title="java.lang.String">String</abbr></code> |
| `doubleFormat` | s | `%.5e` | <code><abbr title="java.lang.String">String</abbr></code> |
| `defaultFunctions` | npm[] | `[ea.f.nOfIterations()]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.util.function.Function">Function</abbr>&lt;? super <abbr title="io.github.ericmedvet.jgea.core.solver.POCPopulationState">POCPopulationState</abbr>&lt;?, G, S, Q, ?&gt;, ?&gt;&gt;</code> |
| `functions` | npm[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.util.function.Function">Function</abbr>&lt;? super <abbr title="io.github.ericmedvet.jgea.core.solver.POCPopulationState">POCPopulationState</abbr>&lt;?, G, S, Q, ?&gt;, ?&gt;&gt;</code> |
| `individualFunctions` | npm[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.util.function.Function">Function</abbr>&lt;? super <abbr title="io.github.ericmedvet.jgea.core.solver.Individual">Individual</abbr>&lt;G, S, Q&gt;, ?&gt;&gt;</code> |
| `defaultRunFunctions` | npm[] | `[ea.f.runKey(key = "run.problem.name"), ea.f.runKey(key = "run.solver.name"), ea.f.runKey(key = "run.randomGenerator.seed")]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.util.function.Function">Function</abbr>&lt;? super <abbr title="io.github.ericmedvet.jgea.experimenter.Run">Run</abbr>&lt;?, G, S, Q&gt;, ?&gt;&gt;</code> |
| `runFunctions` | npm[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.util.function.Function">Function</abbr>&lt;? super <abbr title="io.github.ericmedvet.jgea.experimenter.Run">Run</abbr>&lt;?, G, S, Q&gt;, ?&gt;&gt;</code> |
| `deferred` | b | `false` | <code>boolean</code> |
| `onlyLast` | b | `false` | <code>boolean</code> |
| `condition` | npm | `predicate.always()` | <code><abbr title="java.util.function.Predicate">Predicate</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.experimenter.Run">Run</abbr>&lt;?, G, S, Q&gt;&gt;</code> |

Produces <code><abbr title="java.util.function.BiFunction">BiFunction</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.experimenter.Experiment">Experiment</abbr>, <abbr title="java.util.concurrent.ExecutorService">ExecutorService</abbr>, <abbr title="io.github.ericmedvet.jgea.core.listener.ListenerFactory">ListenerFactory</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.solver.POCPopulationState">POCPopulationState</abbr>&lt;?, G, S, Q, ?&gt;, <abbr title="io.github.ericmedvet.jgea.experimenter.Run">Run</abbr>&lt;?, G, S, Q&gt;&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Listeners.allCsv()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.listener.bestCsv()`

`ea.l.bestCsv(path; errorString; intFormat; doubleFormat; defaultFunctions; functions; defaultRunFunctions; runFunctions; deferred; onlyLast; condition)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `path` | s |  | <code><abbr title="java.lang.String">String</abbr></code> |
| `errorString` | s | `NA` | <code><abbr title="java.lang.String">String</abbr></code> |
| `intFormat` | s | `%d` | <code><abbr title="java.lang.String">String</abbr></code> |
| `doubleFormat` | s | `%.5e` | <code><abbr title="java.lang.String">String</abbr></code> |
| `defaultFunctions` | npm[] | `[ea.f.nOfIterations(), ea.f.nOfEvals(), ea.f.nOfBirths(), ea.f.elapsedSecs(), f.size(of = ea.f.all()), f.size(of = ea.f.firsts()), f.size(of = ea.f.lasts()), f.uniqueness(of = f.each(of = ea.f.all(); mapF = ea.f.genotype())), f.uniqueness(of = f.each(of = ea.f.all(); mapF = ea.f.solution())), f.uniqueness(of = f.each(of = ea.f.all(); mapF = ea.f.quality()))]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.util.function.Function">Function</abbr>&lt;? super <abbr title="io.github.ericmedvet.jgea.core.solver.POCPopulationState">POCPopulationState</abbr>&lt;?, G, S, Q, ?&gt;, ?&gt;&gt;</code> |
| `functions` | npm[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.util.function.Function">Function</abbr>&lt;? super <abbr title="io.github.ericmedvet.jgea.core.solver.POCPopulationState">POCPopulationState</abbr>&lt;?, G, S, Q, ?&gt;, ?&gt;&gt;</code> |
| `defaultRunFunctions` | npm[] | `[ea.f.runKey(key = "run.problem.name"), ea.f.runKey(key = "run.solver.name"), ea.f.runKey(key = "run.randomGenerator.seed")]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.util.function.Function">Function</abbr>&lt;? super <abbr title="io.github.ericmedvet.jgea.experimenter.Run">Run</abbr>&lt;?, G, S, Q&gt;, ?&gt;&gt;</code> |
| `runFunctions` | npm[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.util.function.Function">Function</abbr>&lt;? super <abbr title="io.github.ericmedvet.jgea.experimenter.Run">Run</abbr>&lt;?, G, S, Q&gt;, ?&gt;&gt;</code> |
| `deferred` | b | `false` | <code>boolean</code> |
| `onlyLast` | b | `false` | <code>boolean</code> |
| `condition` | npm | `predicate.always()` | <code><abbr title="java.util.function.Predicate">Predicate</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.experimenter.Run">Run</abbr>&lt;?, G, S, Q&gt;&gt;</code> |

Produces <code><abbr title="java.util.function.BiFunction">BiFunction</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.experimenter.Experiment">Experiment</abbr>, <abbr title="java.util.concurrent.ExecutorService">ExecutorService</abbr>, <abbr title="io.github.ericmedvet.jgea.core.listener.ListenerFactory">ListenerFactory</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.solver.POCPopulationState">POCPopulationState</abbr>&lt;?, G, S, Q, ?&gt;, <abbr title="io.github.ericmedvet.jgea.experimenter.Run">Run</abbr>&lt;?, G, S, Q&gt;&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Listeners.bestCsv()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.listener.console()`

`ea.l.console(defaultFunctions; functions; defaultRunFunctions; runFunctions; deferred; onlyLast; condition; logExceptions)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `defaultFunctions` | npm[] | `[ea.f.nOfIterations(), ea.f.nOfEvals(), ea.f.nOfBirths(), ea.f.elapsedSecs(), f.size(of = ea.f.all()), f.size(of = ea.f.firsts()), f.size(of = ea.f.lasts()), f.uniqueness(of = f.each(of = ea.f.all(); mapF = ea.f.genotype())), f.uniqueness(of = f.each(of = ea.f.all(); mapF = ea.f.solution())), f.uniqueness(of = f.each(of = ea.f.all(); mapF = ea.f.quality()))]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.util.function.Function">Function</abbr>&lt;? super <abbr title="io.github.ericmedvet.jgea.core.solver.POCPopulationState">POCPopulationState</abbr>&lt;?, G, S, Q, ?&gt;, ?&gt;&gt;</code> |
| `functions` | npm[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.util.function.Function">Function</abbr>&lt;? super <abbr title="io.github.ericmedvet.jgea.core.solver.POCPopulationState">POCPopulationState</abbr>&lt;?, G, S, Q, ?&gt;, ?&gt;&gt;</code> |
| `defaultRunFunctions` | npm[] | `[ea.f.runKey(key = "run.problem.name"), ea.f.runKey(key = "run.solver.name"), ea.f.runKey(key = "run.randomGenerator.seed")]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.util.function.Function">Function</abbr>&lt;? super <abbr title="io.github.ericmedvet.jgea.experimenter.Run">Run</abbr>&lt;?, G, S, Q&gt;, ?&gt;&gt;</code> |
| `runFunctions` | npm[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.util.function.Function">Function</abbr>&lt;? super <abbr title="io.github.ericmedvet.jgea.experimenter.Run">Run</abbr>&lt;?, G, S, Q&gt;, ?&gt;&gt;</code> |
| `deferred` | b | `false` | <code>boolean</code> |
| `onlyLast` | b | `false` | <code>boolean</code> |
| `condition` | npm | `predicate.always()` | <code><abbr title="java.util.function.Predicate">Predicate</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.experimenter.Run">Run</abbr>&lt;?, G, S, Q&gt;&gt;</code> |
| `logExceptions` | b | `false` | <code>boolean</code> |

Produces <code><abbr title="java.util.function.BiFunction">BiFunction</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.experimenter.Experiment">Experiment</abbr>, <abbr title="java.util.concurrent.ExecutorService">ExecutorService</abbr>, <abbr title="io.github.ericmedvet.jgea.core.listener.ListenerFactory">ListenerFactory</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.solver.POCPopulationState">POCPopulationState</abbr>&lt;?, G, S, Q, ?&gt;, <abbr title="io.github.ericmedvet.jgea.experimenter.Run">Run</abbr>&lt;?, G, S, Q&gt;&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Listeners.console()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.listener.net()`

`ea.l.net(defaultFunctions; functions; defaultRunFunctions; runFunctions; serverAddress; serverPort; serverKeyFilePath; pollInterval; condition)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `defaultFunctions` | npm[] | `[ea.f.nOfIterations(), ea.f.nOfEvals(), ea.f.nOfBirths(), ea.f.elapsedSecs(), f.size(of = ea.f.all()), f.size(of = ea.f.firsts()), f.size(of = ea.f.lasts()), f.uniqueness(of = f.each(of = ea.f.all(); mapF = ea.f.genotype())), f.uniqueness(of = f.each(of = ea.f.all(); mapF = ea.f.solution())), f.uniqueness(of = f.each(of = ea.f.all(); mapF = ea.f.quality()))]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="io.github.ericmedvet.jnb.datastructure.NamedFunction">NamedFunction</abbr>&lt;? super <abbr title="io.github.ericmedvet.jgea.core.solver.POCPopulationState">POCPopulationState</abbr>&lt;?, G, S, Q, ?&gt;, ?&gt;&gt;</code> |
| `functions` | npm[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="io.github.ericmedvet.jnb.datastructure.NamedFunction">NamedFunction</abbr>&lt;? super <abbr title="io.github.ericmedvet.jgea.core.solver.POCPopulationState">POCPopulationState</abbr>&lt;?, G, S, Q, ?&gt;, ?&gt;&gt;</code> |
| `defaultRunFunctions` | npm[] | `[ea.f.runKey(key = "run.problem.name"), ea.f.runKey(key = "run.solver.name"), ea.f.runKey(key = "run.randomGenerator.seed")]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.util.function.Function">Function</abbr>&lt;? super <abbr title="io.github.ericmedvet.jgea.experimenter.Run">Run</abbr>&lt;?, G, S, Q&gt;, ?&gt;&gt;</code> |
| `runFunctions` | npm[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.util.function.Function">Function</abbr>&lt;? super <abbr title="io.github.ericmedvet.jgea.experimenter.Run">Run</abbr>&lt;?, G, S, Q&gt;, ?&gt;&gt;</code> |
| `serverAddress` | s | `127.0.0.1` | <code><abbr title="java.lang.String">String</abbr></code> |
| `serverPort` | i | `10979` | <code>int</code> |
| `serverKeyFilePath` | s |  | <code><abbr title="java.lang.String">String</abbr></code> |
| `pollInterval` | d | `1.0` | <code>double</code> |
| `condition` | npm | `predicate.always()` | <code><abbr title="java.util.function.Predicate">Predicate</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.experimenter.Run">Run</abbr>&lt;?, G, S, Q&gt;&gt;</code> |

Produces <code><abbr title="java.util.function.BiFunction">BiFunction</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.experimenter.Experiment">Experiment</abbr>, <abbr title="java.util.concurrent.ExecutorService">ExecutorService</abbr>, <abbr title="io.github.ericmedvet.jgea.core.listener.ListenerFactory">ListenerFactory</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.solver.POCPopulationState">POCPopulationState</abbr>&lt;?, G, S, Q, ?&gt;, <abbr title="io.github.ericmedvet.jgea.experimenter.Run">Run</abbr>&lt;?, G, S, Q&gt;&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Listeners.net()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.listener.onExpDone()`

`ea.l.onExpDone(of; preprocessor; consumers; deferred; condition)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm |  | <code><abbr title="io.github.ericmedvet.jgea.core.listener.AccumulatorFactory">AccumulatorFactory</abbr>&lt;E, O, <abbr title="io.github.ericmedvet.jgea.experimenter.Run">Run</abbr>&lt;?, ?, ?, ?&gt;&gt;</code> |
| `preprocessor` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super O, ? extends P&gt;</code> |
| `consumers` | npm[] | `[ea.consumer.deaf()]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="io.github.ericmedvet.jnb.datastructure.TriConsumer">TriConsumer</abbr>&lt;? super P, <abbr title="io.github.ericmedvet.jgea.experimenter.Run">Run</abbr>&lt;?, ?, ?, ?&gt;, <abbr title="io.github.ericmedvet.jgea.experimenter.Experiment">Experiment</abbr>&gt;&gt;</code> |
| `deferred` | b | `false` | <code>boolean</code> |
| `condition` | npm | `predicate.always()` | <code><abbr title="java.util.function.Predicate">Predicate</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.experimenter.Run">Run</abbr>&lt;?, ?, ?, ?&gt;&gt;</code> |

Produces <code><abbr title="java.util.function.BiFunction">BiFunction</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.experimenter.Experiment">Experiment</abbr>, <abbr title="java.util.concurrent.ExecutorService">ExecutorService</abbr>, <abbr title="io.github.ericmedvet.jgea.core.listener.ListenerFactory">ListenerFactory</abbr>&lt;E, <abbr title="io.github.ericmedvet.jgea.experimenter.Run">Run</abbr>&lt;?, ?, ?, ?&gt;&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Listeners.onExpDone()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.listener.onRunDone()`

`ea.l.onRunDone(of; preprocessor; consumers; deferred; condition)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm |  | <code><abbr title="io.github.ericmedvet.jgea.core.listener.AccumulatorFactory">AccumulatorFactory</abbr>&lt;E, O, <abbr title="io.github.ericmedvet.jgea.experimenter.Run">Run</abbr>&lt;?, ?, ?, ?&gt;&gt;</code> |
| `preprocessor` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super O, ? extends P&gt;</code> |
| `consumers` | npm[] | `[ea.consumer.deaf()]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="io.github.ericmedvet.jnb.datastructure.TriConsumer">TriConsumer</abbr>&lt;? super P, <abbr title="io.github.ericmedvet.jgea.experimenter.Run">Run</abbr>&lt;?, ?, ?, ?&gt;, <abbr title="io.github.ericmedvet.jgea.experimenter.Experiment">Experiment</abbr>&gt;&gt;</code> |
| `deferred` | b | `false` | <code>boolean</code> |
| `condition` | npm | `predicate.always()` | <code><abbr title="java.util.function.Predicate">Predicate</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.experimenter.Run">Run</abbr>&lt;?, ?, ?, ?&gt;&gt;</code> |

Produces <code><abbr title="java.util.function.BiFunction">BiFunction</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.experimenter.Experiment">Experiment</abbr>, <abbr title="java.util.concurrent.ExecutorService">ExecutorService</abbr>, <abbr title="io.github.ericmedvet.jgea.core.listener.ListenerFactory">ListenerFactory</abbr>&lt;E, <abbr title="io.github.ericmedvet.jgea.experimenter.Run">Run</abbr>&lt;?, ?, ?, ?&gt;&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Listeners.onRunDone()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.listener.saveForExp()`

`ea.l.saveForExp(of; preprocessor; consumers; deferred; condition; path; processor)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm |  | <code><abbr title="io.github.ericmedvet.jgea.core.listener.AccumulatorFactory">AccumulatorFactory</abbr>&lt;E, O, <abbr title="io.github.ericmedvet.jgea.experimenter.Run">Run</abbr>&lt;?, ?, ?, ?&gt;&gt;</code> |
| `preprocessor` | npm | `null` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super O, ? extends P&gt;</code> |
| `consumers` | npm[] | `[ea.c.saver(path = "../run-{run.index:%04d}")]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="io.github.ericmedvet.jnb.datastructure.TriConsumer">TriConsumer</abbr>&lt;? super P, <abbr title="io.github.ericmedvet.jgea.experimenter.Run">Run</abbr>&lt;?, ?, ?, ?&gt;, <abbr title="io.github.ericmedvet.jgea.experimenter.Experiment">Experiment</abbr>&gt;&gt;</code> |
| `deferred` | b | `false` | <code>boolean</code> |
| `condition` | npm | `predicate.always()` | <code><abbr title="java.util.function.Predicate">Predicate</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.experimenter.Run">Run</abbr>&lt;?, ?, ?, ?&gt;&gt;</code> |
| `path` | s | `../run-{run.index:%04d}` | <code><abbr title="java.lang.String">String</abbr></code> |
| `processor` | npm | `` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="java.util.function.BiFunction">BiFunction</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.experimenter.Experiment">Experiment</abbr>, <abbr title="java.util.concurrent.ExecutorService">ExecutorService</abbr>, <abbr title="io.github.ericmedvet.jgea.core.listener.ListenerFactory">ListenerFactory</abbr>&lt;E, <abbr title="io.github.ericmedvet.jgea.experimenter.Run">Run</abbr>&lt;?, ?, ?, ?&gt;&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Listeners.onExpDone()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.listener.saveForRun()`

`ea.l.saveForRun(of; preprocessor; consumers; deferred; condition; path; processor)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm |  | <code><abbr title="io.github.ericmedvet.jgea.core.listener.AccumulatorFactory">AccumulatorFactory</abbr>&lt;E, O, <abbr title="io.github.ericmedvet.jgea.experimenter.Run">Run</abbr>&lt;?, ?, ?, ?&gt;&gt;</code> |
| `preprocessor` | npm | `null` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super O, ? extends P&gt;</code> |
| `consumers` | npm[] | `[ea.c.saver(path = "run-{run.index:%04d}")]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="io.github.ericmedvet.jnb.datastructure.TriConsumer">TriConsumer</abbr>&lt;? super P, <abbr title="io.github.ericmedvet.jgea.experimenter.Run">Run</abbr>&lt;?, ?, ?, ?&gt;, <abbr title="io.github.ericmedvet.jgea.experimenter.Experiment">Experiment</abbr>&gt;&gt;</code> |
| `deferred` | b | `false` | <code>boolean</code> |
| `condition` | npm | `predicate.always()` | <code><abbr title="java.util.function.Predicate">Predicate</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.experimenter.Run">Run</abbr>&lt;?, ?, ?, ?&gt;&gt;</code> |
| `path` | s | `run-{run.index:%04d}` | <code><abbr title="java.lang.String">String</abbr></code> |
| `processor` | npm | `` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="java.util.function.BiFunction">BiFunction</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.experimenter.Experiment">Experiment</abbr>, <abbr title="java.util.concurrent.ExecutorService">ExecutorService</abbr>, <abbr title="io.github.ericmedvet.jgea.core.listener.ListenerFactory">ListenerFactory</abbr>&lt;E, <abbr title="io.github.ericmedvet.jgea.experimenter.Run">Run</abbr>&lt;?, ?, ?, ?&gt;&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Listeners.onRunDone()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.listener.saveLastPopulationForRun()`

`ea.l.saveLastPopulationForRun(of; preprocessor; consumers; deferred; condition; path; processor)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `ea.acc.lastPopulationMap()` | <code><abbr title="io.github.ericmedvet.jgea.core.listener.AccumulatorFactory">AccumulatorFactory</abbr>&lt;E, O, <abbr title="io.github.ericmedvet.jgea.experimenter.Run">Run</abbr>&lt;?, ?, ?, ?&gt;&gt;</code> |
| `preprocessor` | npm | `null` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super O, ? extends P&gt;</code> |
| `consumers` | npm[] | `[ea.c.saver(path = "run-{run.index:%04d}")]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="io.github.ericmedvet.jnb.datastructure.TriConsumer">TriConsumer</abbr>&lt;? super P, <abbr title="io.github.ericmedvet.jgea.experimenter.Run">Run</abbr>&lt;?, ?, ?, ?&gt;, <abbr title="io.github.ericmedvet.jgea.experimenter.Experiment">Experiment</abbr>&gt;&gt;</code> |
| `deferred` | b | `false` | <code>boolean</code> |
| `condition` | npm | `predicate.always()` | <code><abbr title="java.util.function.Predicate">Predicate</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.experimenter.Run">Run</abbr>&lt;?, ?, ?, ?&gt;&gt;</code> |
| `path` | s | `run-{run.index:%04d}-last-pop` | <code><abbr title="java.lang.String">String</abbr></code> |
| `processor` | npm | `f.identity()` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="java.util.function.BiFunction">BiFunction</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.experimenter.Experiment">Experiment</abbr>, <abbr title="java.util.concurrent.ExecutorService">ExecutorService</abbr>, <abbr title="io.github.ericmedvet.jgea.core.listener.ListenerFactory">ListenerFactory</abbr>&lt;E, <abbr title="io.github.ericmedvet.jgea.experimenter.Run">Run</abbr>&lt;?, ?, ?, ?&gt;&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Listeners.onRunDone()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.listener.savePlotForExp()`

`ea.l.savePlotForExp(of; preprocessor; consumers; deferred; condition; path; processor; plot)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `null` | <code><abbr title="io.github.ericmedvet.jgea.core.listener.AccumulatorFactory">AccumulatorFactory</abbr>&lt;E, O, <abbr title="io.github.ericmedvet.jgea.experimenter.Run">Run</abbr>&lt;?, ?, ?, ?&gt;&gt;</code> |
| `preprocessor` | npm | `null` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super O, ? extends P&gt;</code> |
| `consumers` | npm[] | `[ea.c.saver(path = "../run-{run.index:%04d}")]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="io.github.ericmedvet.jnb.datastructure.TriConsumer">TriConsumer</abbr>&lt;? super P, <abbr title="io.github.ericmedvet.jgea.experimenter.Run">Run</abbr>&lt;?, ?, ?, ?&gt;, <abbr title="io.github.ericmedvet.jgea.experimenter.Experiment">Experiment</abbr>&gt;&gt;</code> |
| `deferred` | b | `false` | <code>boolean</code> |
| `condition` | npm | `predicate.always()` | <code><abbr title="java.util.function.Predicate">Predicate</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.experimenter.Run">Run</abbr>&lt;?, ?, ?, ?&gt;&gt;</code> |
| `path` | s | `../run-{run.index:%04d}` | <code><abbr title="java.lang.String">String</abbr></code> |
| `processor` | npm | `ea.f.imagePlotter()` | <code><abbr title="java.lang.String">String</abbr></code> |
| `plot` | npm | `` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="java.util.function.BiFunction">BiFunction</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.experimenter.Experiment">Experiment</abbr>, <abbr title="java.util.concurrent.ExecutorService">ExecutorService</abbr>, <abbr title="io.github.ericmedvet.jgea.core.listener.ListenerFactory">ListenerFactory</abbr>&lt;E, <abbr title="io.github.ericmedvet.jgea.experimenter.Run">Run</abbr>&lt;?, ?, ?, ?&gt;&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Listeners.onExpDone()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.listener.savePlotForRun()`

`ea.l.savePlotForRun(of; preprocessor; consumers; deferred; condition; path; processor; plot)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `null` | <code><abbr title="io.github.ericmedvet.jgea.core.listener.AccumulatorFactory">AccumulatorFactory</abbr>&lt;E, O, <abbr title="io.github.ericmedvet.jgea.experimenter.Run">Run</abbr>&lt;?, ?, ?, ?&gt;&gt;</code> |
| `preprocessor` | npm | `null` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super O, ? extends P&gt;</code> |
| `consumers` | npm[] | `[ea.c.saver(path = "run-{run.index:%04d}")]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="io.github.ericmedvet.jnb.datastructure.TriConsumer">TriConsumer</abbr>&lt;? super P, <abbr title="io.github.ericmedvet.jgea.experimenter.Run">Run</abbr>&lt;?, ?, ?, ?&gt;, <abbr title="io.github.ericmedvet.jgea.experimenter.Experiment">Experiment</abbr>&gt;&gt;</code> |
| `deferred` | b | `false` | <code>boolean</code> |
| `condition` | npm | `predicate.always()` | <code><abbr title="java.util.function.Predicate">Predicate</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.experimenter.Run">Run</abbr>&lt;?, ?, ?, ?&gt;&gt;</code> |
| `path` | s | `run-{run.index:%04d}` | <code><abbr title="java.lang.String">String</abbr></code> |
| `processor` | npm | `ea.f.imagePlotter()` | <code><abbr title="java.lang.String">String</abbr></code> |
| `plot` | npm | `` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="java.util.function.BiFunction">BiFunction</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.experimenter.Experiment">Experiment</abbr>, <abbr title="java.util.concurrent.ExecutorService">ExecutorService</abbr>, <abbr title="io.github.ericmedvet.jgea.core.listener.ListenerFactory">ListenerFactory</abbr>&lt;E, <abbr title="io.github.ericmedvet.jgea.experimenter.Run">Run</abbr>&lt;?, ?, ?, ?&gt;&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Listeners.onRunDone()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.listener.tui()`

`ea.l.tui(defaultFunctions; functions; defaultRunFunctions; runFunctions; condition)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `defaultFunctions` | npm[] | `[ea.f.nOfIterations(), ea.f.nOfEvals(), ea.f.nOfBirths(), ea.f.elapsedSecs(), f.size(of = ea.f.all()), f.size(of = ea.f.firsts()), f.size(of = ea.f.lasts()), f.uniqueness(of = f.each(of = ea.f.all(); mapF = ea.f.genotype())), f.uniqueness(of = f.each(of = ea.f.all(); mapF = ea.f.solution())), f.uniqueness(of = f.each(of = ea.f.all(); mapF = ea.f.quality()))]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="io.github.ericmedvet.jnb.datastructure.NamedFunction">NamedFunction</abbr>&lt;? super <abbr title="io.github.ericmedvet.jgea.core.solver.POCPopulationState">POCPopulationState</abbr>&lt;?, G, S, Q, ?&gt;, ?&gt;&gt;</code> |
| `functions` | npm[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="io.github.ericmedvet.jnb.datastructure.NamedFunction">NamedFunction</abbr>&lt;? super <abbr title="io.github.ericmedvet.jgea.core.solver.POCPopulationState">POCPopulationState</abbr>&lt;?, G, S, Q, ?&gt;, ?&gt;&gt;</code> |
| `defaultRunFunctions` | npm[] | `[ea.f.runKey(key = "run.problem.name"), ea.f.runKey(key = "run.solver.name"), ea.f.runKey(key = "run.randomGenerator.seed")]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.util.function.Function">Function</abbr>&lt;? super <abbr title="io.github.ericmedvet.jgea.experimenter.Run">Run</abbr>&lt;?, G, S, Q&gt;, ?&gt;&gt;</code> |
| `runFunctions` | npm[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.util.function.Function">Function</abbr>&lt;? super <abbr title="io.github.ericmedvet.jgea.experimenter.Run">Run</abbr>&lt;?, G, S, Q&gt;, ?&gt;&gt;</code> |
| `condition` | npm | `predicate.always()` | <code><abbr title="java.util.function.Predicate">Predicate</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.experimenter.Run">Run</abbr>&lt;?, G, S, Q&gt;&gt;</code> |

Produces <code><abbr title="java.util.function.BiFunction">BiFunction</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.experimenter.Experiment">Experiment</abbr>, <abbr title="java.util.concurrent.ExecutorService">ExecutorService</abbr>, <abbr title="io.github.ericmedvet.jgea.core.listener.ListenerFactory">ListenerFactory</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.solver.POCPopulationState">POCPopulationState</abbr>&lt;?, G, S, Q, ?&gt;, <abbr title="io.github.ericmedvet.jgea.experimenter.Run">Run</abbr>&lt;?, G, S, Q&gt;&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Listeners.tui()` by jgea-experimenter:2.7.1-SNAPSHOT

## Package `ea.mapper`

Aliases: `ea.m`, `ea.mapper`

### Builder `ea.mapper.aggregatedInputNds()`

`ea.m.aggregatedInputNds(of; types; windowT)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `ea.m.identity()` | <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;X, <abbr title="io.github.ericmedvet.jsdynsym.core.numerical.NumericalDynamicalSystem">NumericalDynamicalSystem</abbr>&lt;?&gt;&gt;</code> |
| `types` | e[] | `[CURRENT, TREND]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="io.github.ericmedvet.jsdynsym.core.numerical.AggregatedInput$Type">AggregatedInput$Type</abbr>&gt;</code> |
| `windowT` | d | `1.0` | <code>double</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;X, <abbr title="io.github.ericmedvet.jsdynsym.core.numerical.NumericalDynamicalSystem">NumericalDynamicalSystem</abbr>&lt;?&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Mappers.aggregatedInputNds()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.mapper.bsToGrammarGrid()`

`ea.m.bsToGrammarGrid(of; grammar; l; overwrite; criteria)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `ea.m.identity()` | <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.representation.sequence.bit.BitString">BitString</abbr>&gt;</code> |
| `grammar` | npm |  | <code><abbr title="io.github.ericmedvet.jgea.core.representation.grammar.grid.GridGrammar">GridGrammar</abbr>&lt;T&gt;</code> |
| `l` | i | `256` | <code>int</code> |
| `overwrite` | b | `false` | <code>boolean</code> |
| `criteria` | e[] | `[LEAST_RECENT, LOWEST_Y, LOWEST_X]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.representation.grammar.grid.StandardGridDeveloper$SortingCriterion">StandardGridDeveloper$SortingCriterion</abbr>&gt;</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;X, <abbr title="io.github.ericmedvet.jnb.datastructure.Grid">Grid</abbr>&lt;T&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Mappers.bsToGrammarGrid()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.mapper.dsSplit()`

`ea.m.dsSplit(of)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `ea.m.identity()` | <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;X, <abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.Double">Double</abbr>&gt;&gt;</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;X, <abbr title="io.github.ericmedvet.jnb.datastructure.Pair">Pair</abbr>&lt;<abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.Double">Double</abbr>&gt;, <abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.Double">Double</abbr>&gt;&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Mappers.dsSplit()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.mapper.dsToBitString()`

`ea.m.dsToBitString(of; t)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `ea.m.identity()` | <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;X, <abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.Double">Double</abbr>&gt;&gt;</code> |
| `t` | d | `0.0` | <code>double</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.representation.sequence.bit.BitString">BitString</abbr>&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Mappers.dsToBitString()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.mapper.dsToFixedGrid()`

`ea.m.dsToFixedGrid(of; rate; negItem; posItem)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `ea.m.identity()` | <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;X, <abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.Double">Double</abbr>&gt;&gt;</code> |
| `rate` | d | `0.25` | <code>double</code> |
| `negItem` | npm |  | <code>T</code> |
| `posItem` | npm |  | <code>T</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;X, <abbr title="io.github.ericmedvet.jnb.datastructure.Grid">Grid</abbr>&lt;T&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Mappers.dsToFixedGrid()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.mapper.dsToGrammarGrid()`

`ea.m.dsToGrammarGrid(of; grammar; l; overwrite; criteria)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `ea.m.identity()` | <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;X, <abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.Double">Double</abbr>&gt;&gt;</code> |
| `grammar` | npm |  | <code><abbr title="io.github.ericmedvet.jgea.core.representation.grammar.grid.GridGrammar">GridGrammar</abbr>&lt;T&gt;</code> |
| `l` | i | `256` | <code>int</code> |
| `overwrite` | b | `false` | <code>boolean</code> |
| `criteria` | e[] | `[LEAST_RECENT, LOWEST_Y, LOWEST_X]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.representation.grammar.grid.StandardGridDeveloper$SortingCriterion">StandardGridDeveloper$SortingCriterion</abbr>&gt;</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;X, <abbr title="io.github.ericmedvet.jnb.datastructure.Grid">Grid</abbr>&lt;T&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Mappers.dsToGrammarGrid()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.mapper.dsToIs()`

`ea.m.dsToIs(of; range)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `ea.m.identity()` | <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;X, <abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.Double">Double</abbr>&gt;&gt;</code> |
| `range` | npm | `ds.range(min = -1; max = 1)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.representation.sequence.integer.IntString">IntString</abbr>&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Mappers.dsToIs()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.mapper.dsToNpnds()`

`ea.m.dsToNpnds(of; npnds)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `ea.m.identity()` | <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;X, <abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.Double">Double</abbr>&gt;&gt;</code> |
| `npnds` | npm |  | <code><abbr title="io.github.ericmedvet.jsdynsym.buildable.builders.NumericalDynamicalSystems$Builder">NumericalDynamicalSystems$Builder</abbr>&lt;P, S&gt;</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;X, P&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Mappers.dsToNpnds()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.mapper.dsToThresholdedGrid()`

`ea.m.dsToThresholdedGrid(of; t; negItem; posItem)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `ea.m.identity()` | <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;X, <abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.Double">Double</abbr>&gt;&gt;</code> |
| `t` | d | `0.0` | <code>double</code> |
| `negItem` | npm |  | <code>T</code> |
| `posItem` | npm |  | <code>T</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;X, <abbr title="io.github.ericmedvet.jnb.datastructure.Grid">Grid</abbr>&lt;T&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Mappers.dsToThresholdedGrid()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.mapper.enhancedNds()`

`ea.m.enhancedNds(of; windowT; types)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `ea.m.identity()` | <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;X, <abbr title="io.github.ericmedvet.jsdynsym.core.numerical.NumericalDynamicalSystem">NumericalDynamicalSystem</abbr>&lt;?&gt;&gt;</code> |
| `windowT` | d |  | <code>double</code> |
| `types` | e[] | `[CURRENT, TREND, AVG]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="io.github.ericmedvet.jsdynsym.core.numerical.EnhancedInput$Type">EnhancedInput$Type</abbr>&gt;</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;X, <abbr title="io.github.ericmedvet.jsdynsym.core.numerical.NumericalDynamicalSystem">NumericalDynamicalSystem</abbr>&lt;?&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Mappers.enhancedNds()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.mapper.fGraphToNmrf()`

`ea.m.fGraphToNmrf(of; postOperator)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `ea.m.identity()` | <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.representation.graph.Graph">Graph</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.representation.graph.Node">Node</abbr>, <abbr title="java.lang.Double">Double</abbr>&gt;&gt;</code> |
| `postOperator` | npm | `ds.f.doubleOp(activationF = identity)` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;<abbr title="java.lang.Double">Double</abbr>, <abbr title="java.lang.Double">Double</abbr>&gt;</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.representation.NamedMultivariateRealFunction">NamedMultivariateRealFunction</abbr>&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Mappers.fGraphToNmrf()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.mapper.grammarTreeBP()`

`ea.m.grammarTreeBP(of; problem)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `ea.m.identity()` | <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.representation.tree.Tree">Tree</abbr>&lt;N&gt;&gt;</code> |
| `problem` | npm |  | <code><abbr title="io.github.ericmedvet.jgea.core.representation.grammar.string.GrammarBasedProblem">GrammarBasedProblem</abbr>&lt;N, S&gt;</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;X, S&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Mappers.grammarTreeBP()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.mapper.grammarTreeRegression()`

`ea.m.grammarTreeRegression(of)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `ea.m.identity()` | <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.representation.tree.Tree">Tree</abbr>&lt;<abbr title="java.lang.String">String</abbr>&gt;&gt;</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.representation.tree.Tree">Tree</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.representation.tree.numeric.Element">Element</abbr>&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Mappers.grammarTreeRegression()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.mapper.identity()`

`ea.m.identity()`

Produces <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;X, X&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Mappers.identity()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.mapper.isToGrammarGrid()`

`ea.m.isToGrammarGrid(of; grammar; upperBound; l; overwrite; criteria)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `ea.m.identity()` | <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.representation.sequence.integer.IntString">IntString</abbr>&gt;</code> |
| `grammar` | npm |  | <code><abbr title="io.github.ericmedvet.jgea.core.representation.grammar.grid.GridGrammar">GridGrammar</abbr>&lt;T&gt;</code> |
| `upperBound` | i | `16` | <code>int</code> |
| `l` | i | `256` | <code>int</code> |
| `overwrite` | b | `false` | <code>boolean</code> |
| `criteria` | e[] | `[LEAST_RECENT, LOWEST_Y, LOWEST_X]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.representation.grammar.grid.StandardGridDeveloper$SortingCriterion">StandardGridDeveloper$SortingCriterion</abbr>&gt;</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;X, <abbr title="io.github.ericmedvet.jnb.datastructure.Grid">Grid</abbr>&lt;T&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Mappers.isToGrammarGrid()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.mapper.isToGrid()`

`ea.m.isToGrid(of; items)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `ea.m.identity()` | <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.representation.sequence.integer.IntString">IntString</abbr>&gt;</code> |
| `items` | npm[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;T&gt;</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;X, <abbr title="io.github.ericmedvet.jnb.datastructure.Grid">Grid</abbr>&lt;T&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Mappers.isToGrid()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.mapper.isToString()`

`ea.m.isToString(of; alphabet)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `ea.m.identity()` | <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.representation.sequence.integer.IntString">IntString</abbr>&gt;</code> |
| `alphabet` | s | `abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;X, <abbr title="java.lang.String">String</abbr>&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Mappers.isToString()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.mapper.multiSrTreeToNmrf()`

`ea.m.multiSrTreeToNmrf(of; postOperator)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `ea.m.identity()` | <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;X, <abbr title="java.util.List">List</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.representation.tree.Tree">Tree</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.representation.tree.numeric.Element">Element</abbr>&gt;&gt;&gt;</code> |
| `postOperator` | npm | `ds.f.doubleOp(activationF = identity)` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;<abbr title="java.lang.Double">Double</abbr>, <abbr title="java.lang.Double">Double</abbr>&gt;</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.representation.NamedMultivariateRealFunction">NamedMultivariateRealFunction</abbr>&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Mappers.multiSrTreeToNmrf()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.mapper.nmrfToGrid()`

`ea.m.nmrfToGrid(of; items)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `ea.m.identity()` | <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.representation.NamedMultivariateRealFunction">NamedMultivariateRealFunction</abbr>&gt;</code> |
| `items` | npm[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;T&gt;</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;X, <abbr title="io.github.ericmedvet.jnb.datastructure.Grid">Grid</abbr>&lt;T&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Mappers.nmrfToGrid()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.mapper.nmrfToMrca()`

`ea.m.nmrfToMrca(of; nOfAdditionalChannels; kernels; initializer; range; additiveCoefficient; alivenessThreshold; toroidal)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `ea.m.identity()` | <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.representation.NamedMultivariateRealFunction">NamedMultivariateRealFunction</abbr>&gt;</code> |
| `nOfAdditionalChannels` | i | `1` | <code>int</code> |
| `kernels` | e[] | `[IDENTITY, LAPLACIAN, SOBEL_EDGES]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.problem.ca.MultivariateRealGridCellularAutomaton$Kernel">MultivariateRealGridCellularAutomaton$Kernel</abbr>&gt;</code> |
| `initializer` | e | `CENTER_ALL` | <code><abbr title="io.github.ericmedvet.jgea.problem.ca.MultivariateRealGridCellularAutomaton$Initializer">MultivariateRealGridCellularAutomaton$Initializer</abbr></code> |
| `range` | npm | `m.range(min = -1; max = 1)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |
| `additiveCoefficient` | d | `1.0` | <code>double</code> |
| `alivenessThreshold` | d | `0.0` | <code>double</code> |
| `toroidal` | b | `false` | <code>boolean</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.problem.ca.MultivariateRealGridCellularAutomaton">MultivariateRealGridCellularAutomaton</abbr>&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Mappers.nmrfToMrca()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.mapper.nmrfToNds()`

`ea.m.nmrfToNds(of)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `ea.m.identity()` | <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.representation.NamedMultivariateRealFunction">NamedMultivariateRealFunction</abbr>&gt;</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;X, <abbr title="io.github.ericmedvet.jsdynsym.core.numerical.NumericalDynamicalSystem">NumericalDynamicalSystem</abbr>&lt;?&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Mappers.nmrfToNds()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.mapper.nmrfToNurf()`

`ea.m.nmrfToNurf(of)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `ea.m.identity()` | <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.representation.NamedMultivariateRealFunction">NamedMultivariateRealFunction</abbr>&gt;</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.representation.NamedUnivariateRealFunction">NamedUnivariateRealFunction</abbr>&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Mappers.nmrfToNurf()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.mapper.noisedNds()`

`ea.m.noisedNds(of; inputSigma; outputSigma; randomGenerator)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `ea.m.identity()` | <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;X, <abbr title="io.github.ericmedvet.jsdynsym.core.numerical.NumericalDynamicalSystem">NumericalDynamicalSystem</abbr>&lt;?&gt;&gt;</code> |
| `inputSigma` | d | `0.0` | <code>double</code> |
| `outputSigma` | d | `0.0` | <code>double</code> |
| `randomGenerator` | npm | `m.defaultRG()` | <code><abbr title="java.util.random.RandomGenerator">RandomGenerator</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;X, <abbr title="io.github.ericmedvet.jsdynsym.core.numerical.NumericalDynamicalSystem">NumericalDynamicalSystem</abbr>&lt;?&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Mappers.noisedNds()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.mapper.noisedNmrf()`

`ea.m.noisedNmrf(of; sigma; randomGenerator)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `ea.m.identity()` | <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.representation.NamedMultivariateRealFunction">NamedMultivariateRealFunction</abbr>&gt;</code> |
| `sigma` | d | `0.0` | <code>double</code> |
| `randomGenerator` | npm | `m.defaultRG()` | <code><abbr title="java.util.random.RandomGenerator">RandomGenerator</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.representation.NamedMultivariateRealFunction">NamedMultivariateRealFunction</abbr>&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Mappers.noisedNmrf()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.mapper.ntissToNmrf()`

`ea.m.ntissToNmrf(of)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `ea.m.identity()` | <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;X, <abbr title="io.github.ericmedvet.jsdynsym.core.numerical.NumericalTimeInvariantStatelessSystem">NumericalTimeInvariantStatelessSystem</abbr>&gt;</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.representation.NamedMultivariateRealFunction">NamedMultivariateRealFunction</abbr>&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Mappers.ntissToNmrf()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.mapper.nurfToNds()`

`ea.m.nurfToNds(of)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `ea.m.identity()` | <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.representation.NamedUnivariateRealFunction">NamedUnivariateRealFunction</abbr>&gt;</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;X, <abbr title="io.github.ericmedvet.jsdynsym.core.numerical.NumericalDynamicalSystem">NumericalDynamicalSystem</abbr>&lt;?&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Mappers.nurfToNds()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.mapper.oGraphToNmrf()`

`ea.m.oGraphToNmrf(of; postOperator)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `ea.m.identity()` | <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.representation.graph.Graph">Graph</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.representation.graph.Node">Node</abbr>, <abbr title="io.github.ericmedvet.jgea.core.representation.graph.numeric.operatorgraph.OperatorGraph$NonValuedArc">OperatorGraph$NonValuedArc</abbr>&gt;&gt;</code> |
| `postOperator` | npm | `ds.f.doubleOp(activationF = identity)` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;<abbr title="java.lang.Double">Double</abbr>, <abbr title="java.lang.Double">Double</abbr>&gt;</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.representation.NamedMultivariateRealFunction">NamedMultivariateRealFunction</abbr>&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Mappers.oGraphToNmrf()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.mapper.pair()`

`ea.m.pair(of; first; second)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `ea.m.identity()` | <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;X, <abbr title="io.github.ericmedvet.jnb.datastructure.Pair">Pair</abbr>&lt;F1, S1&gt;&gt;</code> |
| `first` | npm | `ea.m.identity()` | <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;F1, F2&gt;</code> |
| `second` | npm | `ea.m.identity()` | <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;S1, S2&gt;</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;X, <abbr title="io.github.ericmedvet.jnb.datastructure.Pair">Pair</abbr>&lt;F2, S2&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Mappers.pair()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.mapper.srTreeToNurf()`

`ea.m.srTreeToNurf(of; postOperator)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `ea.m.identity()` | <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.representation.tree.Tree">Tree</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.representation.tree.numeric.Element">Element</abbr>&gt;&gt;</code> |
| `postOperator` | npm | `ds.f.doubleOp(activationF = identity)` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;<abbr title="java.lang.Double">Double</abbr>, <abbr title="java.lang.Double">Double</abbr>&gt;</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.representation.NamedUnivariateRealFunction">NamedUnivariateRealFunction</abbr>&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Mappers.srTreeToNurf()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.mapper.steppedNds()`

`ea.m.steppedNds(of; stepT)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `ea.m.identity()` | <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;X, <abbr title="io.github.ericmedvet.jsdynsym.core.numerical.NumericalDynamicalSystem">NumericalDynamicalSystem</abbr>&lt;?&gt;&gt;</code> |
| `stepT` | d | `1.0` | <code>double</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;X, <abbr title="io.github.ericmedvet.jsdynsym.core.numerical.NumericalDynamicalSystem">NumericalDynamicalSystem</abbr>&lt;?&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Mappers.steppedNds()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.mapper.ttpnToProgram()`

`ea.m.ttpnToProgram(of; maxNOfSteps; maxNOfTokens; maxTokensSize; maxSingleTokenSize; skipBlocked)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `ea.m.identity()` | <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.representation.programsynthesis.ttpn.Network">Network</abbr>&gt;</code> |
| `maxNOfSteps` | i | `128` | <code>int</code> |
| `maxNOfTokens` | i | `256` | <code>int</code> |
| `maxTokensSize` | i | `1024` | <code>int</code> |
| `maxSingleTokenSize` | i | `128` | <code>int</code> |
| `skipBlocked` | b | `true` | <code>boolean</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;X, <abbr title="io.github.ericmedvet.jgea.core.representation.programsynthesis.Program">Program</abbr>&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Mappers.ttpnToProgram()` by jgea-experimenter:2.7.1-SNAPSHOT

## Package `ea.misc`

### Builder `ea.misc.caVideo()`

`ea.misc.caVideo(gray; caStateRange; nOfSteps; sizeRate; marginRate; frameRate; fontSize)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `gray` | b | `true` | <code>boolean</code> |
| `caStateRange` | npm | `m.range(min = -1; max = 1)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |
| `nOfSteps` | i | `100` | <code>int</code> |
| `sizeRate` | i | `10` | <code>int</code> |
| `marginRate` | d | `0.0` | <code>double</code> |
| `frameRate` | d | `10.0` | <code>double</code> |
| `fontSize` | d | `10.0` | <code>double</code> |

Produces <code><abbr title="io.github.ericmedvet.jviz.core.drawer.VideoBuilder">VideoBuilder</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.problem.ca.MultivariateRealGridCellularAutomaton">MultivariateRealGridCellularAutomaton</abbr>&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Miscs.caVideo()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.misc.ch()`

`ea.misc.ch(s)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `s` | s |  | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="java.lang.Character">Character</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Miscs.ch()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.misc.colorByName()`

`ea.misc.colorByName(name)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `name` | s |  | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="java.awt.Color">Color</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Miscs.colorByName()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.misc.colorByRgb()`

`ea.misc.colorByRgb(r; g; b)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `r` | i |  | <code>int</code> |
| `g` | i |  | <code>int</code> |
| `b` | i |  | <code>int</code> |

Produces <code><abbr title="java.awt.Color">Color</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Miscs.colorByRgb()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.misc.entry()`

`ea.misc.entry(key; value)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `key` | npm |  | <code>K</code> |
| `value` | npm |  | <code>V</code> |

Produces <code><abbr title="java.util.Map$Entry">Map$Entry</abbr>&lt;K, V&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Miscs.entry()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.misc.imgByName()`

`ea.misc.imgByName(name; gateBGColor; w; h; marginRate)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `name` | s |  | <code><abbr title="java.lang.String">String</abbr></code> |
| `gateBGColor` | npm | `ea.misc.colorByName(name = black)` | <code><abbr title="java.awt.Color">Color</abbr></code> |
| `w` | i | `15` | <code>int</code> |
| `h` | i | `15` | <code>int</code> |
| `marginRate` | d | `0.1` | <code>double</code> |

Produces <code><abbr title="java.awt.image.BufferedImage">BufferedImage</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Miscs.imgByName()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.misc.imgFromString()`

`ea.misc.imgFromString(s; borderColor; gateBGColor; w; h; marginRate)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `s` | s |  | <code><abbr title="java.lang.String">String</abbr></code> |
| `borderColor` | npm | `ea.misc.colorByName(name = white)` | <code><abbr title="java.awt.Color">Color</abbr></code> |
| `gateBGColor` | npm | `ea.misc.colorByName(name = black)` | <code><abbr title="java.awt.Color">Color</abbr></code> |
| `w` | i | `159` | <code>int</code> |
| `h` | i | `15` | <code>int</code> |
| `marginRate` | d | `0.1` | <code>double</code> |

Produces <code><abbr title="java.awt.image.BufferedImage">BufferedImage</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Miscs.imgFromString()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.misc.lossyAverage()`

`ea.misc.lossyAverage(memoryFactor)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `memoryFactor` | d | `0.5` | <code>double</code> |

Produces <code><abbr title="java.util.function.BinaryOperator">BinaryOperator</abbr>&lt;<abbr title="java.lang.Double">Double</abbr>&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Miscs.lossyAverage()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.misc.map()`

`ea.misc.map(entries)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `entries` | npm[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.util.Map$Entry">Map$Entry</abbr>&lt;K, V&gt;&gt;</code> |

Produces <code><abbr title="java.util.Map">Map</abbr>&lt;K, V&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Miscs.map()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.misc.mapFromLists()`

`ea.misc.mapFromLists(keys; values)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `keys` | npm[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;K&gt;</code> |
| `values` | npm[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;V&gt;</code> |

Produces <code><abbr title="java.util.Map">Map</abbr>&lt;K, V&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Miscs.mapFromLists()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.misc.sEntry()`

`ea.misc.sEntry(key; value)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `key` | s |  | <code><abbr title="java.lang.String">String</abbr></code> |
| `value` | npm |  | <code>V</code> |

Produces <code><abbr title="java.util.Map$Entry">Map$Entry</abbr>&lt;<abbr title="java.lang.String">String</abbr>, V&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Miscs.sEntry()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.misc.sMapFromLists()`

`ea.misc.sMapFromLists(keys; values)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `keys` | s[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.String">String</abbr>&gt;</code> |
| `values` | npm[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;V&gt;</code> |

Produces <code><abbr title="java.util.Map">Map</abbr>&lt;<abbr title="java.lang.String">String</abbr>, V&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Miscs.sMapFromLists()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.misc.toVideo()`

`ea.misc.toVideo(drawer)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `drawer` | npm |  | <code><abbr title="io.github.ericmedvet.jsdynsym.control.SimulationOutcomeDrawer">SimulationOutcomeDrawer</abbr>&lt;S&gt;</code> |

Produces <code><abbr title="io.github.ericmedvet.jviz.core.drawer.VideoBuilder">VideoBuilder</abbr>&lt;<abbr title="io.github.ericmedvet.jsdynsym.control.Simulation$Outcome">Simulation$Outcome</abbr>&lt;S&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Miscs.toVideo()` by jgea-experimenter:2.7.1-SNAPSHOT

## Package `ea.plot.multi`

Aliases: `ea.plot.m`, `ea.plot.multi`

### Builder `ea.plot.multi.quality()`

`ea.plot.m.quality(xSubplot; ySubplot; line; x; y; valueAggregator; minAggregator; maxAggregator; xRange; yRange; xQuantization; q)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `xSubplot` | npm | `ea.f.runString(s = "_"; name = none)` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super R, <abbr title="java.lang.String">String</abbr>&gt;</code> |
| `ySubplot` | npm | `ea.f.runString(s = "{run.problem.name}"; name = problem)` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super R, <abbr title="java.lang.String">String</abbr>&gt;</code> |
| `line` | npm | `ea.f.runString(s = "{run.solver.name}"; name = solver)` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super R, <abbr title="java.lang.String">String</abbr>&gt;</code> |
| `x` | npm | `f.quantized(q = 1; of = ea.f.nOfEvals())` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super E, ? extends <abbr title="java.lang.Number">Number</abbr>&gt;</code> |
| `y` | npm | `f.composition(of = ea.f.quality(of = ea.f.best()); then = f.identity())` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super E, ? extends <abbr title="java.lang.Number">Number</abbr>&gt;</code> |
| `valueAggregator` | npm | `f.median()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;<abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.Number">Number</abbr>&gt;, <abbr title="java.lang.Number">Number</abbr>&gt;</code> |
| `minAggregator` | npm | `f.percentile(p = 25)` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;<abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.Number">Number</abbr>&gt;, <abbr title="java.lang.Number">Number</abbr>&gt;</code> |
| `maxAggregator` | npm | `f.percentile(p = 75)` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;<abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.Number">Number</abbr>&gt;, <abbr title="java.lang.Number">Number</abbr>&gt;</code> |
| `xRange` | npm | `m.range(min = -Infinity; max = Infinity)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |
| `yRange` | npm | `m.range(min = -Infinity; max = Infinity)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |
| `xQuantization` | i | `1` | <code><abbr title="java.lang.String">String</abbr></code> |
| `q` | npm | `f.identity()` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.experimenter.listener.plot.AggregatedXYDataSeriesMRPAF">AggregatedXYDataSeriesMRPAF</abbr>&lt;E, R, <abbr title="java.lang.String">String</abbr>&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.MultiPlots.xy()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.plot.multi.qualityBoxplot()`

`ea.plot.m.qualityBoxplot(xSubplot; ySubplot; box; y; predicateValue; condition; yRange; q)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `xSubplot` | npm | `ea.f.runString(s = "_"; name = none)` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super R, <abbr title="java.lang.String">String</abbr>&gt;</code> |
| `ySubplot` | npm | `ea.f.runString(s = "{run.problem.name}"; name = problem)` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super R, <abbr title="java.lang.String">String</abbr>&gt;</code> |
| `box` | npm | `ea.f.runString(s = "{run.solver.name}"; name = solver)` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super R, <abbr title="java.lang.String">String</abbr>&gt;</code> |
| `y` | npm | `f.composition(of = ea.f.quality(of = ea.f.best()); then = f.identity())` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super E, ? extends <abbr title="java.lang.Number">Number</abbr>&gt;</code> |
| `predicateValue` | npm | `ea.f.rate(of = ea.f.progress())` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;E, X&gt;</code> |
| `condition` | npm | `predicate.gtEq(t = 1)` | <code><abbr title="java.util.function.Predicate">Predicate</abbr>&lt;X&gt;</code> |
| `yRange` | npm | `m.range(min = -Infinity; max = Infinity)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |
| `q` | npm | `f.identity()` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.experimenter.listener.plot.DistributionMRPAF">DistributionMRPAF</abbr>&lt;E, R, <abbr title="java.lang.String">String</abbr>, X&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.MultiPlots.yBoxplot()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.plot.multi.uniqueness()`

`ea.plot.m.uniqueness(xSubplot; ySubplot; line; x; y; valueAggregator; minAggregator; maxAggregator; xRange; yRange; xQuantization)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `xSubplot` | npm | `ea.f.runString(s = "_"; name = none)` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super R, <abbr title="java.lang.String">String</abbr>&gt;</code> |
| `ySubplot` | npm | `ea.f.runString(s = "{run.problem.name}"; name = problem)` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super R, <abbr title="java.lang.String">String</abbr>&gt;</code> |
| `line` | npm | `ea.f.runString(s = "{run.solver.name}"; name = solver)` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super R, <abbr title="java.lang.String">String</abbr>&gt;</code> |
| `x` | npm | `f.quantized(q = 1; of = ea.f.nOfEvals())` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super E, ? extends <abbr title="java.lang.Number">Number</abbr>&gt;</code> |
| `y` | npm | `f.uniqueness(of = f.each(of = ea.f.all(); mapF = ea.f.genotype()))` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super E, ? extends <abbr title="java.lang.Number">Number</abbr>&gt;</code> |
| `valueAggregator` | npm | `f.median()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;<abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.Number">Number</abbr>&gt;, <abbr title="java.lang.Number">Number</abbr>&gt;</code> |
| `minAggregator` | npm | `f.percentile(p = 25)` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;<abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.Number">Number</abbr>&gt;, <abbr title="java.lang.Number">Number</abbr>&gt;</code> |
| `maxAggregator` | npm | `f.percentile(p = 75)` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;<abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.Number">Number</abbr>&gt;, <abbr title="java.lang.Number">Number</abbr>&gt;</code> |
| `xRange` | npm | `m.range(min = -Infinity; max = Infinity)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |
| `yRange` | npm | `m.range(min = -Infinity; max = Infinity)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |
| `xQuantization` | i | `1` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.experimenter.listener.plot.AggregatedXYDataSeriesMRPAF">AggregatedXYDataSeriesMRPAF</abbr>&lt;E, R, <abbr title="java.lang.String">String</abbr>&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.MultiPlots.xy()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.plot.multi.uniquenessBoxplot()`

`ea.plot.m.uniquenessBoxplot(xSubplot; ySubplot; box; y; predicateValue; condition; yRange)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `xSubplot` | npm | `ea.f.runString(s = "_"; name = none)` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super R, <abbr title="java.lang.String">String</abbr>&gt;</code> |
| `ySubplot` | npm | `ea.f.runString(s = "{run.problem.name}"; name = problem)` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super R, <abbr title="java.lang.String">String</abbr>&gt;</code> |
| `box` | npm | `ea.f.runString(s = "{run.solver.name}"; name = solver)` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super R, <abbr title="java.lang.String">String</abbr>&gt;</code> |
| `y` | npm | `f.uniqueness(of = f.each(of = ea.f.all(); mapF = ea.f.genotype()))` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super E, ? extends <abbr title="java.lang.Number">Number</abbr>&gt;</code> |
| `predicateValue` | npm | `ea.f.rate(of = ea.f.progress())` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;E, X&gt;</code> |
| `condition` | npm | `predicate.gtEq(t = 1)` | <code><abbr title="java.util.function.Predicate">Predicate</abbr>&lt;X&gt;</code> |
| `yRange` | npm | `m.range(min = -Infinity; max = Infinity)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.experimenter.listener.plot.DistributionMRPAF">DistributionMRPAF</abbr>&lt;E, R, <abbr title="java.lang.String">String</abbr>, X&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.MultiPlots.yBoxplot()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.plot.multi.xy()`

`ea.plot.m.xy(xSubplot; ySubplot; line; x; y; valueAggregator; minAggregator; maxAggregator; xRange; yRange)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `xSubplot` | npm |  | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super R, <abbr title="java.lang.String">String</abbr>&gt;</code> |
| `ySubplot` | npm |  | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super R, <abbr title="java.lang.String">String</abbr>&gt;</code> |
| `line` | npm |  | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super R, <abbr title="java.lang.String">String</abbr>&gt;</code> |
| `x` | npm |  | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super E, ? extends <abbr title="java.lang.Number">Number</abbr>&gt;</code> |
| `y` | npm |  | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super E, ? extends <abbr title="java.lang.Number">Number</abbr>&gt;</code> |
| `valueAggregator` | npm | `f.median()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;<abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.Number">Number</abbr>&gt;, <abbr title="java.lang.Number">Number</abbr>&gt;</code> |
| `minAggregator` | npm | `f.percentile(p = 25)` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;<abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.Number">Number</abbr>&gt;, <abbr title="java.lang.Number">Number</abbr>&gt;</code> |
| `maxAggregator` | npm | `f.percentile(p = 75)` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;<abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.Number">Number</abbr>&gt;, <abbr title="java.lang.Number">Number</abbr>&gt;</code> |
| `xRange` | npm | `m.range(min = -Infinity; max = Infinity)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |
| `yRange` | npm | `m.range(min = -Infinity; max = Infinity)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.experimenter.listener.plot.AggregatedXYDataSeriesMRPAF">AggregatedXYDataSeriesMRPAF</abbr>&lt;E, R, <abbr title="java.lang.String">String</abbr>&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.MultiPlots.xy()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.plot.multi.xyExp()`

`ea.plot.m.xyExp(xSubplot; ySubplot; line; x; y; valueAggregator; minAggregator; maxAggregator; xRange; yRange; xQuantization)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `xSubplot` | npm | `ea.f.runString(s = "_"; name = none)` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super R, <abbr title="java.lang.String">String</abbr>&gt;</code> |
| `ySubplot` | npm | `ea.f.runString(s = "{run.problem.name}"; name = problem)` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super R, <abbr title="java.lang.String">String</abbr>&gt;</code> |
| `line` | npm | `ea.f.runString(s = "{run.solver.name}"; name = solver)` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super R, <abbr title="java.lang.String">String</abbr>&gt;</code> |
| `x` | npm | `f.quantized(q = 1; of = ea.f.nOfEvals())` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super E, ? extends <abbr title="java.lang.Number">Number</abbr>&gt;</code> |
| `y` | npm |  | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super E, ? extends <abbr title="java.lang.Number">Number</abbr>&gt;</code> |
| `valueAggregator` | npm | `f.median()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;<abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.Number">Number</abbr>&gt;, <abbr title="java.lang.Number">Number</abbr>&gt;</code> |
| `minAggregator` | npm | `f.percentile(p = 25)` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;<abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.Number">Number</abbr>&gt;, <abbr title="java.lang.Number">Number</abbr>&gt;</code> |
| `maxAggregator` | npm | `f.percentile(p = 75)` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;<abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.Number">Number</abbr>&gt;, <abbr title="java.lang.Number">Number</abbr>&gt;</code> |
| `xRange` | npm | `m.range(min = -Infinity; max = Infinity)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |
| `yRange` | npm | `m.range(min = -Infinity; max = Infinity)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |
| `xQuantization` | i | `1` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.experimenter.listener.plot.AggregatedXYDataSeriesMRPAF">AggregatedXYDataSeriesMRPAF</abbr>&lt;E, R, <abbr title="java.lang.String">String</abbr>&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.MultiPlots.xy()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.plot.multi.yBoxplot()`

`ea.plot.m.yBoxplot(xSubplot; ySubplot; box; y; predicateValue; condition; yRange)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `xSubplot` | npm |  | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super R, <abbr title="java.lang.String">String</abbr>&gt;</code> |
| `ySubplot` | npm |  | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super R, <abbr title="java.lang.String">String</abbr>&gt;</code> |
| `box` | npm |  | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super R, <abbr title="java.lang.String">String</abbr>&gt;</code> |
| `y` | npm |  | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super E, ? extends <abbr title="java.lang.Number">Number</abbr>&gt;</code> |
| `predicateValue` | npm |  | <code><abbr title="java.util.function.Function">Function</abbr>&lt;E, X&gt;</code> |
| `condition` | npm | `predicate.gtEq(t = 1)` | <code><abbr title="java.util.function.Predicate">Predicate</abbr>&lt;X&gt;</code> |
| `yRange` | npm | `m.range(min = -Infinity; max = Infinity)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.experimenter.listener.plot.DistributionMRPAF">DistributionMRPAF</abbr>&lt;E, R, <abbr title="java.lang.String">String</abbr>, X&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.MultiPlots.yBoxplot()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.plot.multi.yBoxplotExp()`

`ea.plot.m.yBoxplotExp(xSubplot; ySubplot; box; y; predicateValue; condition; yRange)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `xSubplot` | npm | `ea.f.runString(s = "_"; name = none)` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super R, <abbr title="java.lang.String">String</abbr>&gt;</code> |
| `ySubplot` | npm | `ea.f.runString(s = "{run.problem.name}"; name = problem)` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super R, <abbr title="java.lang.String">String</abbr>&gt;</code> |
| `box` | npm | `ea.f.runString(s = "{run.solver.name}"; name = solver)` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super R, <abbr title="java.lang.String">String</abbr>&gt;</code> |
| `y` | npm |  | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super E, ? extends <abbr title="java.lang.Number">Number</abbr>&gt;</code> |
| `predicateValue` | npm | `ea.f.rate(of = ea.f.progress())` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;E, X&gt;</code> |
| `condition` | npm | `predicate.gtEq(t = 1)` | <code><abbr title="java.util.function.Predicate">Predicate</abbr>&lt;X&gt;</code> |
| `yRange` | npm | `m.range(min = -Infinity; max = Infinity)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.experimenter.listener.plot.DistributionMRPAF">DistributionMRPAF</abbr>&lt;E, R, <abbr title="java.lang.String">String</abbr>, X&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.MultiPlots.yBoxplot()` by jgea-experimenter:2.7.1-SNAPSHOT

## Package `ea.plot.single`

Aliases: `ea.plot.s`, `ea.plot.single`

### Builder `ea.plot.single.biObjectivePopulation()`

`ea.plot.s.biObjectivePopulation(title; points; x; y; predicateValue; unique; condition; xRange; yRange)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `title` | npm | `ea.f.runString(s = "Fronts with {run.solver.name} on {run.problem.name} (seed={run.randomGenerator.seed})"; name = title)` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super R, <abbr title="java.lang.String">String</abbr>&gt;</code> |
| `points` | npm[] | `[ea.f.firsts(), ea.f.mids(), ea.f.lasts()]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.util.function.Function">Function</abbr>&lt;? super E, <abbr title="java.util.Collection">Collection</abbr>&lt;P&gt;&gt;&gt;</code> |
| `x` | npm | `f.nThMapValue(of = ea.f.quality(); n = 0)` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super P, ? extends <abbr title="java.lang.Number">Number</abbr>&gt;</code> |
| `y` | npm | `f.nThMapValue(of = ea.f.quality(); n = 1)` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super P, ? extends <abbr title="java.lang.Number">Number</abbr>&gt;</code> |
| `predicateValue` | npm | `f.quantized(q = 0.05; of = ea.f.rate(of = ea.f.progress()); format = "%.2f")` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;E, X&gt;</code> |
| `unique` | b | `true` | <code>boolean</code> |
| `condition` | npm | `predicate.inD(values = [0.0; 0.1; 0.25; 0.5; 1.0])` | <code><abbr title="java.util.function.Predicate">Predicate</abbr>&lt;X&gt;</code> |
| `xRange` | npm | `m.range(min = -Infinity; max = Infinity)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |
| `yRange` | npm | `m.range(min = -Infinity; max = Infinity)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.experimenter.listener.plot.XYDataSeriesSEPAF">XYDataSeriesSEPAF</abbr>&lt;E, R, X, P&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.SinglePlots.xyes()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.plot.single.coMe()`

`ea.plot.s.coMe(title; values; grids; predicateValue; condition; valueRange; unique; q)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `title` | npm | `ea.f.runString(s = "Archives of {run.solver.name} on {run.problem.name} (seed={run.randomGenerator.seed})"; name = title)` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super R, <abbr title="java.lang.String">String</abbr>&gt;</code> |
| `values` | npm[] | `[f.composition(of = ea.f.quality(); then = f.identity())]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.util.function.Function">Function</abbr>&lt;? super G, ? extends <abbr title="java.lang.Number">Number</abbr>&gt;&gt;</code> |
| `grids` | npm[] | `[ea.f.archiveToGrid(of = ea.f.coMeArchive1()), ea.f.archiveToGrid(of = ea.f.coMeArchive2())]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.util.function.Function">Function</abbr>&lt;? super E, <abbr title="io.github.ericmedvet.jnb.datastructure.Grid">Grid</abbr>&lt;G&gt;&gt;&gt;</code> |
| `predicateValue` | npm | `f.quantized(q = 0.05; of = ea.f.rate(of = ea.f.progress()); format = "%.2f")` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;E, X&gt;</code> |
| `condition` | npm | `predicate.inD(values = [0.0; 0.1; 0.25; 0.5; 1.0])` | <code><abbr title="java.util.function.Predicate">Predicate</abbr>&lt;X&gt;</code> |
| `valueRange` | npm | `m.range(min = -Infinity; max = Infinity)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |
| `unique` | b | `true` | <code>boolean</code> |
| `q` | npm | `f.identity()` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.experimenter.listener.plot.UnivariateGridSEPAF">UnivariateGridSEPAF</abbr>&lt;E, R, X, G&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.SinglePlots.grid()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.plot.single.coMeStrategies()`

`ea.plot.s.coMeStrategies(title; fields; pointPairs; predicateValue; condition; unique)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `title` | npm | `ea.f.runString(s = "Strategies (2D fields) of {run.solver.name} on {run.problem.name} (seed={runrandomGenerator.seed})"; name = title)` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super R, <abbr title="java.lang.String">String</abbr>&gt;</code> |
| `fields` | npm[] | `[ea.f.coMeStrategy1Field(), ea.f.coMeStrategy2Field()]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.util.function.Function">Function</abbr>&lt;? super E, F&gt;&gt;</code> |
| `pointPairs` | npm[] | `[f.identity()]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.util.function.Function">Function</abbr>&lt;? super F, ? extends <abbr title="java.util.Map">Map</abbr>&lt;<abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.Double">Double</abbr>&gt;, <abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.Double">Double</abbr>&gt;&gt;&gt;&gt;</code> |
| `predicateValue` | npm | `f.quantized(q = 0.05; of = ea.f.rate(of = ea.f.progress()); format = "%.2f")` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;E, X&gt;</code> |
| `condition` | npm | `predicate.inD(values = [0.0; 0.1; 0.25; 0.5; 1.0])` | <code><abbr title="java.util.function.Predicate">Predicate</abbr>&lt;X&gt;</code> |
| `unique` | b | `true` | <code>boolean</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.experimenter.listener.plot.VectorialFieldSEPAF">VectorialFieldSEPAF</abbr>&lt;E, R, X, F&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.SinglePlots.field()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.plot.single.field()`

`ea.plot.s.field(title; fields; pointPairs; predicateValue; condition; unique)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `title` | npm |  | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super R, <abbr title="java.lang.String">String</abbr>&gt;</code> |
| `fields` | npm[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.util.function.Function">Function</abbr>&lt;? super E, F&gt;&gt;</code> |
| `pointPairs` | npm[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.util.function.Function">Function</abbr>&lt;? super F, ? extends <abbr title="java.util.Map">Map</abbr>&lt;<abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.Double">Double</abbr>&gt;, <abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.Double">Double</abbr>&gt;&gt;&gt;&gt;</code> |
| `predicateValue` | npm |  | <code><abbr title="java.util.function.Function">Function</abbr>&lt;E, X&gt;</code> |
| `condition` | npm | `predicate.ltEq(t = 1)` | <code><abbr title="java.util.function.Predicate">Predicate</abbr>&lt;X&gt;</code> |
| `unique` | b | `true` | <code>boolean</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.experimenter.listener.plot.VectorialFieldSEPAF">VectorialFieldSEPAF</abbr>&lt;E, R, X, F&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.SinglePlots.field()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.plot.single.fieldRun()`

`ea.plot.s.fieldRun(title; fields; pointPairs; predicateValue; condition; unique)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `title` | npm | `ea.f.runString(s = "{run.solver.name} on {run.problem.name} (seed={run.randomGenerator.seed})"; name = title)` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super R, <abbr title="java.lang.String">String</abbr>&gt;</code> |
| `fields` | npm[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.util.function.Function">Function</abbr>&lt;? super E, F&gt;&gt;</code> |
| `pointPairs` | npm[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.util.function.Function">Function</abbr>&lt;? super F, ? extends <abbr title="java.util.Map">Map</abbr>&lt;<abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.Double">Double</abbr>&gt;, <abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.Double">Double</abbr>&gt;&gt;&gt;&gt;</code> |
| `predicateValue` | npm | `f.quantized(q = 0.05; of = ea.f.rate(of = ea.f.progress()); format = "%.2f")` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;E, X&gt;</code> |
| `condition` | npm | `predicate.inD(values = [0.0; 0.1; 0.25; 0.5; 1.0])` | <code><abbr title="java.util.function.Predicate">Predicate</abbr>&lt;X&gt;</code> |
| `unique` | b | `true` | <code>boolean</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.experimenter.listener.plot.VectorialFieldSEPAF">VectorialFieldSEPAF</abbr>&lt;E, R, X, F&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.SinglePlots.field()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.plot.single.grid()`

`ea.plot.s.grid(title; values; grids; predicateValue; condition; valueRange; unique)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `title` | npm |  | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super R, <abbr title="java.lang.String">String</abbr>&gt;</code> |
| `values` | npm[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.util.function.Function">Function</abbr>&lt;? super G, ? extends <abbr title="java.lang.Number">Number</abbr>&gt;&gt;</code> |
| `grids` | npm[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.util.function.Function">Function</abbr>&lt;? super E, <abbr title="io.github.ericmedvet.jnb.datastructure.Grid">Grid</abbr>&lt;G&gt;&gt;&gt;</code> |
| `predicateValue` | npm |  | <code><abbr title="java.util.function.Function">Function</abbr>&lt;E, X&gt;</code> |
| `condition` | npm | `predicate.ltEq(t = 1)` | <code><abbr title="java.util.function.Predicate">Predicate</abbr>&lt;X&gt;</code> |
| `valueRange` | npm | `m.range(min = -Infinity; max = Infinity)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |
| `unique` | b | `true` | <code>boolean</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.experimenter.listener.plot.UnivariateGridSEPAF">UnivariateGridSEPAF</abbr>&lt;E, R, X, G&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.SinglePlots.grid()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.plot.single.gridRun()`

`ea.plot.s.gridRun(title; values; grids; predicateValue; condition; valueRange; unique)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `title` | npm | `ea.f.runString(s = "{run.solver.name} on {run.problem.name} (seed={run.randomGenerator.seed})"; name = title)` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super R, <abbr title="java.lang.String">String</abbr>&gt;</code> |
| `values` | npm[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.util.function.Function">Function</abbr>&lt;? super G, ? extends <abbr title="java.lang.Number">Number</abbr>&gt;&gt;</code> |
| `grids` | npm[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.util.function.Function">Function</abbr>&lt;? super E, <abbr title="io.github.ericmedvet.jnb.datastructure.Grid">Grid</abbr>&lt;G&gt;&gt;&gt;</code> |
| `predicateValue` | npm | `f.quantized(q = 0.05; of = ea.f.rate(of = ea.f.progress()); format = "%.2f")` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;E, X&gt;</code> |
| `condition` | npm | `predicate.inD(values = [0.0; 0.1; 0.25; 0.5; 1.0])` | <code><abbr title="java.util.function.Predicate">Predicate</abbr>&lt;X&gt;</code> |
| `valueRange` | npm | `m.range(min = -Infinity; max = Infinity)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |
| `unique` | b | `true` | <code>boolean</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.experimenter.listener.plot.UnivariateGridSEPAF">UnivariateGridSEPAF</abbr>&lt;E, R, X, G&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.SinglePlots.grid()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.plot.single.gridState()`

`ea.plot.s.gridState(title; values; grids; predicateValue; condition; valueRange; unique; q)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `title` | npm | `ea.f.runString(s = "Grid population of {run.solver.name} on {run.problem.name} (seed={run.randomGenerator.seed})"; name = title)` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super R, <abbr title="java.lang.String">String</abbr>&gt;</code> |
| `values` | npm[] | `[f.composition(of = ea.f.quality(); then = f.identity())]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.util.function.Function">Function</abbr>&lt;? super G, ? extends <abbr title="java.lang.Number">Number</abbr>&gt;&gt;</code> |
| `grids` | npm[] | `[ea.f.stateGrid()]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.util.function.Function">Function</abbr>&lt;? super E, <abbr title="io.github.ericmedvet.jnb.datastructure.Grid">Grid</abbr>&lt;G&gt;&gt;&gt;</code> |
| `predicateValue` | npm | `f.quantized(q = 0.05; of = ea.f.rate(of = ea.f.progress()); format = "%.2f")` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;E, X&gt;</code> |
| `condition` | npm | `predicate.inD(values = [0.0; 0.1; 0.25; 0.5; 1.0])` | <code><abbr title="java.util.function.Predicate">Predicate</abbr>&lt;X&gt;</code> |
| `valueRange` | npm | `m.range(min = -Infinity; max = Infinity)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |
| `unique` | b | `true` | <code>boolean</code> |
| `q` | npm | `f.identity()` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.experimenter.listener.plot.UnivariateGridSEPAF">UnivariateGridSEPAF</abbr>&lt;E, R, X, G&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.SinglePlots.grid()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.plot.single.landscape()`

`ea.plot.s.landscape(title; predicateValue; condition; mapper; q; xRange; yRange; xF; yF; valueRange; unique)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `title` | npm | `ea.f.runString(s = "{run.solver.name} on {run.problem.name} (seed={run.randomGenerator.seed})"; name = title)` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super <abbr title="io.github.ericmedvet.jgea.experimenter.Run">Run</abbr>&lt;?, <abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.Double">Double</abbr>&gt;, S, Q&gt;, <abbr title="java.lang.String">String</abbr>&gt;</code> |
| `predicateValue` | npm | `f.quantized(q = 0.05; of = ea.f.rate(of = ea.f.progress()); format = "%.2f")` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.solver.POCPopulationState">POCPopulationState</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.solver.Individual">Individual</abbr>&lt;<abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.Double">Double</abbr>&gt;, S, Q&gt;, <abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.Double">Double</abbr>&gt;, S, Q, P&gt;, X&gt;</code> |
| `condition` | npm | `predicate.inD(values = [0.0; 0.1; 0.25; 0.5; 1.0])` | <code><abbr title="java.util.function.Predicate">Predicate</abbr>&lt;X&gt;</code> |
| `mapper` | npm | `ea.m.identity()` | <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;<abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.Double">Double</abbr>&gt;, S&gt;</code> |
| `q` | npm | `ea.m.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;Q, <abbr title="java.lang.Double">Double</abbr>&gt;</code> |
| `xRange` | npm | `m.range(min = -Infinity; max = Infinity)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |
| `yRange` | npm | `m.range(min = -Infinity; max = Infinity)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |
| `xF` | npm | `f.nTh(of = ea.f.genotype(); n = 0)` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.solver.Individual">Individual</abbr>&lt;<abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.Double">Double</abbr>&gt;, S, Q&gt;, <abbr title="java.lang.Double">Double</abbr>&gt;</code> |
| `yF` | npm | `f.nTh(of = ea.f.genotype(); n = 1)` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.solver.Individual">Individual</abbr>&lt;<abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.Double">Double</abbr>&gt;, S, Q&gt;, <abbr title="java.lang.Double">Double</abbr>&gt;</code> |
| `valueRange` | npm | `m.range(min = -Infinity; max = Infinity)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |
| `unique` | b | `true` | <code>boolean</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.experimenter.listener.plot.LandscapeSEPAF">LandscapeSEPAF</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.solver.POCPopulationState">POCPopulationState</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.solver.Individual">Individual</abbr>&lt;<abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.Double">Double</abbr>&gt;, S, Q&gt;, <abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.Double">Double</abbr>&gt;, S, Q, P&gt;, <abbr title="io.github.ericmedvet.jgea.experimenter.Run">Run</abbr>&lt;?, <abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.Double">Double</abbr>&gt;, S, Q&gt;, X, <abbr title="io.github.ericmedvet.jgea.core.solver.Individual">Individual</abbr>&lt;<abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.Double">Double</abbr>&gt;, S, Q&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.SinglePlots.landscape()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.plot.single.maMe2()`

`ea.plot.s.maMe2(title; values; grids; predicateValue; condition; valueRange; unique; q)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `title` | npm | `ea.f.runString(s = "Archives of {run.solver.name} on {run.problem.name} (seed={run.randomGenerator.seed})"; name = title)` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super R, <abbr title="java.lang.String">String</abbr>&gt;</code> |
| `values` | npm[] | `[f.composition(of = ea.f.quality(); then = f.identity())]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.util.function.Function">Function</abbr>&lt;? super G, ? extends <abbr title="java.lang.Number">Number</abbr>&gt;&gt;</code> |
| `grids` | npm[] | `[ea.f.archiveToGrid(of = ea.f.maMeArchive(n = 0)), ea.f.archiveToGrid(of = ea.f.maMeArchive(n = 1))]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.util.function.Function">Function</abbr>&lt;? super E, <abbr title="io.github.ericmedvet.jnb.datastructure.Grid">Grid</abbr>&lt;G&gt;&gt;&gt;</code> |
| `predicateValue` | npm | `f.quantized(q = 0.05; of = ea.f.rate(of = ea.f.progress()); format = "%.2f")` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;E, X&gt;</code> |
| `condition` | npm | `predicate.inD(values = [0.0; 0.1; 0.25; 0.5; 1.0])` | <code><abbr title="java.util.function.Predicate">Predicate</abbr>&lt;X&gt;</code> |
| `valueRange` | npm | `m.range(min = -Infinity; max = Infinity)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |
| `unique` | b | `true` | <code>boolean</code> |
| `q` | npm | `f.identity()` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.experimenter.listener.plot.UnivariateGridSEPAF">UnivariateGridSEPAF</abbr>&lt;E, R, X, G&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.SinglePlots.grid()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.plot.single.me()`

`ea.plot.s.me(title; values; grids; predicateValue; condition; valueRange; unique; q)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `title` | npm | `ea.f.runString(s = "Archive of {run.solver.name} on {run.problem.name} (seed={run.randomGenerator.seed})"; name = title)` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super R, <abbr title="java.lang.String">String</abbr>&gt;</code> |
| `values` | npm[] | `[f.composition(of = ea.f.quality(); then = f.identity())]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.util.function.Function">Function</abbr>&lt;? super G, ? extends <abbr title="java.lang.Number">Number</abbr>&gt;&gt;</code> |
| `grids` | npm[] | `[ea.f.archiveToGrid(of = ea.f.meArchive())]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.util.function.Function">Function</abbr>&lt;? super E, <abbr title="io.github.ericmedvet.jnb.datastructure.Grid">Grid</abbr>&lt;G&gt;&gt;&gt;</code> |
| `predicateValue` | npm | `f.quantized(q = 0.05; of = ea.f.rate(of = ea.f.progress()); format = "%.2f")` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;E, X&gt;</code> |
| `condition` | npm | `predicate.inD(values = [0.0; 0.1; 0.25; 0.5; 1.0])` | <code><abbr title="java.util.function.Predicate">Predicate</abbr>&lt;X&gt;</code> |
| `valueRange` | npm | `m.range(min = -Infinity; max = Infinity)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |
| `unique` | b | `true` | <code>boolean</code> |
| `q` | npm | `f.identity()` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.experimenter.listener.plot.UnivariateGridSEPAF">UnivariateGridSEPAF</abbr>&lt;E, R, X, G&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.SinglePlots.grid()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.plot.single.populationValidation()`

`ea.plot.s.populationValidation(title; points; x; y; predicateValue; unique; condition; xRange; yRange; q; v)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `title` | npm | `ea.f.runString(s = "Population validation of {run.solver.name} on {run.problem.name} (seed={run.randomGenerator.seed})"; name = title)` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super R, <abbr title="java.lang.String">String</abbr>&gt;</code> |
| `points` | npm[] | `[ea.f.all()]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.util.function.Function">Function</abbr>&lt;? super E, <abbr title="java.util.Collection">Collection</abbr>&lt;P&gt;&gt;&gt;</code> |
| `x` | npm | `f.composition(of = ea.f.quality(); then = f.identity())` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super P, ? extends <abbr title="java.lang.Number">Number</abbr>&gt;</code> |
| `y` | npm | `f.composition(of = f.composition(of = ea.f.solution(); then = null); then = f.identity())` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super P, ? extends <abbr title="java.lang.Number">Number</abbr>&gt;</code> |
| `predicateValue` | npm | `f.quantized(q = 0.05; of = ea.f.rate(of = ea.f.progress()); format = "%.2f")` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;E, X&gt;</code> |
| `unique` | b | `true` | <code>boolean</code> |
| `condition` | npm | `predicate.inD(values = [0.0; 0.1; 0.25; 0.5; 1.0])` | <code><abbr title="java.util.function.Predicate">Predicate</abbr>&lt;X&gt;</code> |
| `xRange` | npm | `m.range(min = -Infinity; max = Infinity)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |
| `yRange` | npm | `m.range(min = -Infinity; max = Infinity)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |
| `q` | npm | `f.identity()` | <code><abbr title="java.lang.String">String</abbr></code> |
| `v` | npm | `` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.experimenter.listener.plot.XYDataSeriesSEPAF">XYDataSeriesSEPAF</abbr>&lt;E, R, X, P&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.SinglePlots.xyes()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.plot.single.quality()`

`ea.plot.s.quality(title; x; ys; xRange; yRange; q)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `title` | npm | `ea.f.runString(s = "{run.solver.name} on {run.problem.name} (seed={run.randomGenerator.seed})"; name = title)` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super R, <abbr title="java.lang.String">String</abbr>&gt;</code> |
| `x` | npm | `ea.f.nOfEvals()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super E, ? extends <abbr title="java.lang.Number">Number</abbr>&gt;</code> |
| `ys` | npm[] | `[f.composition(of = ea.f.quality(of = ea.f.best()); then = f.identity())]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.util.function.Function">Function</abbr>&lt;? super E, ? extends <abbr title="java.lang.Number">Number</abbr>&gt;&gt;</code> |
| `xRange` | npm | `m.range(min = -Infinity; max = Infinity)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |
| `yRange` | npm | `m.range(min = -Infinity; max = Infinity)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |
| `q` | npm | `f.identity()` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.experimenter.listener.plot.XYDataSeriesSRPAF">XYDataSeriesSRPAF</abbr>&lt;E, R&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.SinglePlots.xyrs()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.plot.single.uniqueness()`

`ea.plot.s.uniqueness(title; x; ys; xRange; yRange)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `title` | npm | `ea.f.runString(s = "{run.solver.name} on {run.problem.name} (seed={run.randomGenerator.seed})"; name = title)` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super R, <abbr title="java.lang.String">String</abbr>&gt;</code> |
| `x` | npm | `ea.f.nOfEvals()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super E, ? extends <abbr title="java.lang.Number">Number</abbr>&gt;</code> |
| `ys` | npm[] | `[f.uniqueness(of = f.each(of = ea.f.all(); mapF = ea.f.genotype())), f.uniqueness(of = f.each(of = ea.f.all(); mapF = ea.f.solution())), f.uniqueness(of = f.each(of = ea.f.all(); mapF = ea.f.quality()))]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.util.function.Function">Function</abbr>&lt;? super E, ? extends <abbr title="java.lang.Number">Number</abbr>&gt;&gt;</code> |
| `xRange` | npm | `m.range(min = -Infinity; max = Infinity)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |
| `yRange` | npm | `m.range(min = -Infinity; max = Infinity)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.experimenter.listener.plot.XYDataSeriesSRPAF">XYDataSeriesSRPAF</abbr>&lt;E, R&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.SinglePlots.xyrs()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.plot.single.xyes()`

`ea.plot.s.xyes(title; points; x; y; predicateValue; unique; condition; xRange; yRange)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `title` | npm |  | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super R, <abbr title="java.lang.String">String</abbr>&gt;</code> |
| `points` | npm[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.util.function.Function">Function</abbr>&lt;? super E, <abbr title="java.util.Collection">Collection</abbr>&lt;P&gt;&gt;&gt;</code> |
| `x` | npm |  | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super P, ? extends <abbr title="java.lang.Number">Number</abbr>&gt;</code> |
| `y` | npm |  | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super P, ? extends <abbr title="java.lang.Number">Number</abbr>&gt;</code> |
| `predicateValue` | npm |  | <code><abbr title="java.util.function.Function">Function</abbr>&lt;E, X&gt;</code> |
| `unique` | b | `true` | <code>boolean</code> |
| `condition` | npm | `predicate.ltEq(t = 1)` | <code><abbr title="java.util.function.Predicate">Predicate</abbr>&lt;X&gt;</code> |
| `xRange` | npm | `m.range(min = -Infinity; max = Infinity)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |
| `yRange` | npm | `m.range(min = -Infinity; max = Infinity)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.experimenter.listener.plot.XYDataSeriesSEPAF">XYDataSeriesSEPAF</abbr>&lt;E, R, X, P&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.SinglePlots.xyes()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.plot.single.xyrs()`

`ea.plot.s.xyrs(title; x; ys; xRange; yRange)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `title` | npm |  | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super R, <abbr title="java.lang.String">String</abbr>&gt;</code> |
| `x` | npm |  | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super E, ? extends <abbr title="java.lang.Number">Number</abbr>&gt;</code> |
| `ys` | npm[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.util.function.Function">Function</abbr>&lt;? super E, ? extends <abbr title="java.lang.Number">Number</abbr>&gt;&gt;</code> |
| `xRange` | npm | `m.range(min = -Infinity; max = Infinity)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |
| `yRange` | npm | `m.range(min = -Infinity; max = Infinity)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.experimenter.listener.plot.XYDataSeriesSRPAF">XYDataSeriesSRPAF</abbr>&lt;E, R&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.SinglePlots.xyrs()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.plot.single.xyrsRun()`

`ea.plot.s.xyrsRun(title; x; ys; xRange; yRange)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `title` | npm | `ea.f.runString(s = "{run.solver.name} on {run.problem.name} (seed={run.randomGenerator.seed})"; name = title)` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super R, <abbr title="java.lang.String">String</abbr>&gt;</code> |
| `x` | npm | `ea.f.nOfEvals()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;? super E, ? extends <abbr title="java.lang.Number">Number</abbr>&gt;</code> |
| `ys` | npm[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.util.function.Function">Function</abbr>&lt;? super E, ? extends <abbr title="java.lang.Number">Number</abbr>&gt;&gt;</code> |
| `xRange` | npm | `m.range(min = -Infinity; max = Infinity)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |
| `yRange` | npm | `m.range(min = -Infinity; max = Infinity)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.experimenter.listener.plot.XYDataSeriesSRPAF">XYDataSeriesSRPAF</abbr>&lt;E, R&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.SinglePlots.xyrs()` by jgea-experimenter:2.7.1-SNAPSHOT

## Package `ea.problem`

Aliases: `ea.p`, `ea.problem`

### Builder `ea.problem.biSimToBqb()`

`ea.p.biSimToBqb(name; simulation; cFunction; type; qFunction1; qFunction2)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `name` | s | interpolate `{simulation.name}` | <code><abbr title="java.lang.String">String</abbr></code> |
| `simulation` | npm |  | <code><abbr title="io.github.ericmedvet.jsdynsym.control.HomogeneousBiSimulation">HomogeneousBiSimulation</abbr>&lt;S, BS, B&gt;</code> |
| `cFunction` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;Q, C&gt;</code> |
| `type` | e | `MINIMIZE` | <code><abbr title="io.github.ericmedvet.jgea.experimenter.builders.Problems$OptimizationType">Problems$OptimizationType</abbr></code> |
| `qFunction1` | npm |  | <code><abbr title="java.util.function.Function">Function</abbr>&lt;B, Q&gt;</code> |
| `qFunction2` | npm |  | <code><abbr title="java.util.function.Function">Function</abbr>&lt;B, Q&gt;</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.core.problem.TotalOrderQualityBasedBiProblem">TotalOrderQualityBasedBiProblem</abbr>&lt;S, B, Q&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Problems.biSimToBqb()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.problem.biSimToQb()`

`ea.p.biSimToQb(name; simulation; cFunction; type; qFunction; trainingOpponent)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `name` | s | interpolate `{simulation.name}` | <code><abbr title="java.lang.String">String</abbr></code> |
| `simulation` | npm |  | <code><abbr title="io.github.ericmedvet.jsdynsym.control.HomogeneousBiSimulation">HomogeneousBiSimulation</abbr>&lt;S, BS, B&gt;</code> |
| `cFunction` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;Q, C&gt;</code> |
| `type` | e | `MINIMIZE` | <code><abbr title="io.github.ericmedvet.jgea.experimenter.builders.Problems$OptimizationType">Problems$OptimizationType</abbr></code> |
| `qFunction` | npm |  | <code><abbr title="java.util.function.Function">Function</abbr>&lt;B, Q&gt;</code> |
| `trainingOpponent` | npm |  | <code><abbr title="java.util.function.Supplier">Supplier</abbr>&lt;S&gt;</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.core.problem.TotalOrderQualityBasedProblem">TotalOrderQualityBasedProblem</abbr>&lt;S, Q&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Problems.biSimToQb()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.problem.moToSo()`

`ea.p.moToSo(name; objective; moProblem)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `name` | s | interpolate `{moProblem.name}[{objective}]` | <code><abbr title="java.lang.String">String</abbr></code> |
| `objective` | s |  | <code><abbr title="java.lang.String">String</abbr></code> |
| `moProblem` | npm |  | <code><abbr title="io.github.ericmedvet.jgea.core.problem.MultiObjectiveProblem">MultiObjectiveProblem</abbr>&lt;S, Q, O&gt;</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.core.problem.TotalOrderQualityBasedProblem">TotalOrderQualityBasedProblem</abbr>&lt;S, Q&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Problems.moToSo()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.problem.mtToMo()`

`ea.p.mtToMo(name; mtProblem)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `name` | s | interpolate `mt2mo[{mtProblem.name}]` | <code><abbr title="java.lang.String">String</abbr></code> |
| `mtProblem` | npm |  | <code><abbr title="io.github.ericmedvet.jgea.core.problem.MultiTargetProblem">MultiTargetProblem</abbr>&lt;S&gt;</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.core.problem.SimpleMOProblem">SimpleMOProblem</abbr>&lt;S, <abbr title="java.lang.Double">Double</abbr>&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Problems.mtToMo()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.problem.simToSbbmo()`

`ea.p.simToSbbmo(name; simulation; toMinObjectives; toMaxObjectives)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `name` | s | interpolate `{simulation.name}` | <code><abbr title="java.lang.String">String</abbr></code> |
| `simulation` | npm |  | <code><abbr title="io.github.ericmedvet.jsdynsym.control.Simulation">Simulation</abbr>&lt;S, BS, B&gt;</code> |
| `toMinObjectives` | npm[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.util.function.Function">Function</abbr>&lt;B, O&gt;&gt;</code> |
| `toMaxObjectives` | npm[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.util.function.Function">Function</abbr>&lt;B, O&gt;&gt;</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.core.problem.SimpleBBMOProblem">SimpleBBMOProblem</abbr>&lt;S, B, O&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Problems.simToSbbmo()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.problem.simToSmo()`

`ea.p.simToSmo(name; simulation; toMinObjectives; toMaxObjectives)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `name` | s | interpolate `{simulation.name}` | <code><abbr title="java.lang.String">String</abbr></code> |
| `simulation` | npm |  | <code><abbr title="io.github.ericmedvet.jsdynsym.control.Simulation">Simulation</abbr>&lt;S, BS, B&gt;</code> |
| `toMinObjectives` | npm[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.util.function.Function">Function</abbr>&lt;B, O&gt;&gt;</code> |
| `toMaxObjectives` | npm[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.util.function.Function">Function</abbr>&lt;B, O&gt;&gt;</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.core.problem.SimpleMOProblem">SimpleMOProblem</abbr>&lt;S, O&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Problems.simToSmo()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.problem.smoToSubsettedSmo()`

`ea.p.smoToSubsettedSmo(name; objectives; smoProblem)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `name` | s | interpolate `{smoProblem.name}` | <code><abbr title="java.lang.String">String</abbr></code> |
| `objectives` | s[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.String">String</abbr>&gt;</code> |
| `smoProblem` | npm |  | <code><abbr title="io.github.ericmedvet.jgea.core.problem.SimpleMOProblem">SimpleMOProblem</abbr>&lt;S, O&gt;</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.core.problem.SimpleMOProblem">SimpleMOProblem</abbr>&lt;S, O&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Problems.smoToSubsettedSmo()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.problem.totalOrder()`

`ea.p.totalOrder(name; qFunction; cFunction; type)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `name` | s | `{qFunction}` | <code><abbr title="java.lang.String">String</abbr></code> |
| `qFunction` | npm |  | <code><abbr title="java.util.function.Function">Function</abbr>&lt;S, Q&gt;</code> |
| `cFunction` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;Q, C&gt;</code> |
| `type` | e | `MINIMIZE` | <code><abbr title="io.github.ericmedvet.jgea.experimenter.builders.Problems$OptimizationType">Problems$OptimizationType</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.core.problem.TotalOrderQualityBasedProblem">TotalOrderQualityBasedProblem</abbr>&lt;S, Q&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Problems.totalOrder()` by jgea-experimenter:2.7.1-SNAPSHOT

## Package `ea.problem.multivariateRegression`

Aliases: `ea.p.mr`, `ea.p.multivariateRegression`, `ea.problem.mr`, `ea.problem.multivariateRegression`

### Builder `ea.problem.multivariateRegression.fromData()`

`ea.p.mr.fromData(provider; metrics; nFolds; testFold)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `provider` | npm |  | <code><abbr title="io.github.ericmedvet.jgea.core.util.IndexedProvider">IndexedProvider</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.problem.ExampleBasedProblem$Example">ExampleBasedProblem$Example</abbr>&lt;<abbr title="java.util.Map">Map</abbr>&lt;<abbr title="java.lang.String">String</abbr>, <abbr title="java.lang.Double">Double</abbr>&gt;, <abbr title="java.util.Map">Map</abbr>&lt;<abbr title="java.lang.String">String</abbr>, <abbr title="java.lang.Double">Double</abbr>&gt;&gt;&gt;</code> |
| `metrics` | e[] | `[mse]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.problem.regression.univariate.UnivariateRegressionProblem$Metric">UnivariateRegressionProblem$Metric</abbr>&gt;</code> |
| `nFolds` | i | `10` | <code>int</code> |
| `testFold` | i | `0` | <code>int</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.problem.regression.multivariate.MultivariateRegressionProblem">MultivariateRegressionProblem</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.MultivariateRegressionProblems.fromData()` by jgea-experimenter:2.7.1-SNAPSHOT

## Package `ea.problem.programSynthesis`

Aliases: `ea.p.programSynthesis`, `ea.p.ps`, `ea.problem.programSynthesis`, `ea.problem.ps`

### Builder `ea.problem.programSynthesis.synthetic()`

`ea.p.ps.synthetic(name; metrics; maxDissimilarity; randomGenerator; nOfCases; nOfValidationCases; maxExceptionRate; ints; reals; strings; intRange; realRange; stringLengthRange; sequenceSizeRange)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `name` | s |  | <code><abbr title="java.lang.String">String</abbr></code> |
| `metrics` | e[] | `[fail_rate]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.problem.programsynthesis.ProgramSynthesisProblem$Metric">ProgramSynthesisProblem$Metric</abbr>&gt;</code> |
| `maxDissimilarity` | d | `1000.0` | <code>double</code> |
| `randomGenerator` | npm | `m.defaultRG()` | <code><abbr title="java.util.random.RandomGenerator">RandomGenerator</abbr></code> |
| `nOfCases` | i | `20` | <code>int</code> |
| `nOfValidationCases` | i | `100` | <code>int</code> |
| `maxExceptionRate` | d | `0.1` | <code>double</code> |
| `ints` | i[] | `[1, 2, 3, 5, 10]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.Integer">Integer</abbr>&gt;</code> |
| `reals` | d[] | `[1.0, 2.0, 3.0, 1.5, 2.5, 3.14]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.Double">Double</abbr>&gt;</code> |
| `strings` | s[] | `[cat, dog, Hello World!, mummy]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.String">String</abbr>&gt;</code> |
| `intRange` | npm | `m.range(min = -10; max = 100)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |
| `realRange` | npm | `m.range(min = -10; max = 10)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |
| `stringLengthRange` | npm | `m.range(min = 2; max = 20)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |
| `sequenceSizeRange` | npm | `m.range(min = 1; max = 8)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.problem.programsynthesis.ProgramSynthesisProblem">ProgramSynthesisProblem</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.ProgramSynthesisProblems.synthetic()` by jgea-experimenter:2.7.1-SNAPSHOT

## Package `ea.problem.synthetic`

Aliases: `ea.p.s`, `ea.p.synthetic`, `ea.problem.s`, `ea.problem.synthetic`

### Builder `ea.problem.synthetic.ackley()`

`ea.p.s.ackley(name; p)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `name` | s | interpolate `ackley-{p}` | <code><abbr title="java.lang.String">String</abbr></code> |
| `p` | i | `100` | <code>int</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.problem.synthetic.numerical.Ackley">Ackley</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.SyntheticProblems.ackley()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.problem.synthetic.bentCigar()`

`ea.p.s.bentCigar(name; p)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `name` | s | interpolate `bentCigar-{p}` | <code><abbr title="java.lang.String">String</abbr></code> |
| `p` | i | `100` | <code>int</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.problem.synthetic.numerical.BentCigar">BentCigar</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.SyntheticProblems.bentCigar()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.problem.synthetic.charShapeApproximation()`

`ea.p.s.charShapeApproximation(name; target; translation; smoothed; weighted)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `name` | s | `shape-{target}` | <code><abbr title="java.lang.String">String</abbr></code> |
| `target` | s |  | <code><abbr title="java.lang.String">String</abbr></code> |
| `translation` | b | `true` | <code>boolean</code> |
| `smoothed` | b | `true` | <code>boolean</code> |
| `weighted` | b | `true` | <code>boolean</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.problem.grid.CharShapeApproximation">CharShapeApproximation</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.SyntheticProblems.charShapeApproximation()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.problem.synthetic.circularPointsAiming()`

`ea.p.s.circularPointsAiming(name; p; n; radius; center; seed)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `name` | s | interpolate `circularPointsAiming-{p}-{n}` | <code><abbr title="java.lang.String">String</abbr></code> |
| `p` | i | `100` | <code>int</code> |
| `n` | i | `5` | <code>int</code> |
| `radius` | d | `0.5` | <code>double</code> |
| `center` | d | `1.0` | <code>double</code> |
| `seed` | i | `1` | <code>int</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.problem.synthetic.numerical.CircularPointsAiming">CircularPointsAiming</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.SyntheticProblems.circularPointsAiming()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.problem.synthetic.discus()`

`ea.p.s.discus(name; p)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `name` | s | interpolate `discus-{p}` | <code><abbr title="java.lang.String">String</abbr></code> |
| `p` | i | `100` | <code>int</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.problem.synthetic.numerical.Discus">Discus</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.SyntheticProblems.discus()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.problem.synthetic.gaussianMixture2D()`

`ea.p.s.gaussianMixture2D(name; targets; c)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `name` | s | `gm2D` | <code><abbr title="java.lang.String">String</abbr></code> |
| `targets` | d[] | `[-3.0, -2.0, 2.0, 2.0, 2.0, 1.0]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.Double">Double</abbr>&gt;</code> |
| `c` | d | `1.0` | <code>double</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.problem.synthetic.numerical.GaussianMixture2D">GaussianMixture2D</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.SyntheticProblems.gaussianMixture2D()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.problem.synthetic.grammarText()`

`ea.p.s.grammarText(name; target)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `name` | s | interpolate `s({target})` | <code><abbr title="java.lang.String">String</abbr></code> |
| `target` | s | `The white dog!` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.problem.synthetic.Text">Text</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.SyntheticProblems.grammarText()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.problem.synthetic.highConditionedElliptic()`

`ea.p.s.highConditionedElliptic(name; p)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `name` | s | interpolate `highConditionedElliptic-{p}` | <code><abbr title="java.lang.String">String</abbr></code> |
| `p` | i | `100` | <code>int</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.problem.synthetic.numerical.HighConditionedElliptic">HighConditionedElliptic</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.SyntheticProblems.highConditionedElliptic()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.problem.synthetic.intOneMax()`

`ea.p.s.intOneMax(name; p; upperBound)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `name` | s | interpolate `iOneMax-{p}` | <code><abbr title="java.lang.String">String</abbr></code> |
| `p` | i | `100` | <code>int</code> |
| `upperBound` | i | `100` | <code>int</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.problem.synthetic.IntOneMax">IntOneMax</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.SyntheticProblems.intOneMax()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.problem.synthetic.lettersMax()`

`ea.p.s.lettersMax(name; letters; l)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `name` | s | interpolate `lettersMax-{l}-{letters}` | <code><abbr title="java.lang.String">String</abbr></code> |
| `letters` | s[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.String">String</abbr>&gt;</code> |
| `l` | i | `32` | <code>int</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.problem.synthetic.LettersMax">LettersMax</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.SyntheticProblems.lettersMax()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.problem.synthetic.linearPoints()`

`ea.p.s.linearPoints(name; p)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `name` | s | interpolate `lPoints-{p}` | <code><abbr title="java.lang.String">String</abbr></code> |
| `p` | i | `100` | <code>int</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.problem.synthetic.numerical.LinearPoints">LinearPoints</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.SyntheticProblems.linearPoints()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.problem.synthetic.mrCaMorphogenesis()`

`ea.p.s.mrCaMorphogenesis(name; target; gray; fromStep; toStep; caStateRange; targetRange)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `name` | s | interpolate `ca-target-[{minConvergenceStep}-{maxConvergenceStep}]` | <code><abbr title="java.lang.String">String</abbr></code> |
| `target` | npm |  | <code><abbr title="java.awt.image.BufferedImage">BufferedImage</abbr></code> |
| `gray` | b | `true` | <code>boolean</code> |
| `fromStep` | i | `40` | <code>int</code> |
| `toStep` | i | `60` | <code>int</code> |
| `caStateRange` | npm | `m.range(min = -1; max = 1)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |
| `targetRange` | npm | `m.range(min = 0; max = 1)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.problem.ca.MRCAMorphogenesis">MRCAMorphogenesis</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.SyntheticProblems.mrCaMorphogenesis()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.problem.synthetic.mrCaNamedImageMorphogenesis()`

`ea.p.s.mrCaNamedImageMorphogenesis(name; target; gray; fromStep; toStep; caStateRange; targetRange; iName; w; h)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `name` | s | `ca-nImg` | <code><abbr title="java.lang.String">String</abbr></code> |
| `target` | npm | `ea.misc.imgByName(w = 15; h = 15; name = null)` | <code><abbr title="java.awt.image.BufferedImage">BufferedImage</abbr></code> |
| `gray` | b | `false` | <code>boolean</code> |
| `fromStep` | i | `40` | <code>int</code> |
| `toStep` | i | `60` | <code>int</code> |
| `caStateRange` | npm | `m.range(min = -1; max = 1)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |
| `targetRange` | npm | `m.range(min = 0; max = 1)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |
| `iName` | s | `` | <code><abbr title="java.lang.String">String</abbr></code> |
| `w` | i | `15` | <code><abbr title="java.lang.String">String</abbr></code> |
| `h` | i | `15` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.problem.ca.MRCAMorphogenesis">MRCAMorphogenesis</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.SyntheticProblems.mrCaMorphogenesis()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.problem.synthetic.mrCaStringMorphogenesis()`

`ea.p.s.mrCaStringMorphogenesis(name; target; gray; fromStep; toStep; caStateRange; targetRange; s; w; h)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `name` | s | `ca-string` | <code><abbr title="java.lang.String">String</abbr></code> |
| `target` | npm | `ea.misc.imgFromString(s = x; w = 15; h = 15)` | <code><abbr title="java.awt.image.BufferedImage">BufferedImage</abbr></code> |
| `gray` | b | `true` | <code>boolean</code> |
| `fromStep` | i | `40` | <code>int</code> |
| `toStep` | i | `60` | <code>int</code> |
| `caStateRange` | npm | `m.range(min = -1; max = 1)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |
| `targetRange` | npm | `m.range(min = 0; max = 1)` | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |
| `s` | s | `x` | <code><abbr title="java.lang.String">String</abbr></code> |
| `w` | i | `15` | <code><abbr title="java.lang.String">String</abbr></code> |
| `h` | i | `15` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.problem.ca.MRCAMorphogenesis">MRCAMorphogenesis</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.SyntheticProblems.mrCaMorphogenesis()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.problem.synthetic.multiModalIntOneMax()`

`ea.p.s.multiModalIntOneMax(name; p; upperBound; nOfTargets)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `name` | s | interpolate `mmIOneMax-{p}-{nOfTargets}` | <code><abbr title="java.lang.String">String</abbr></code> |
| `p` | i | `100` | <code>int</code> |
| `upperBound` | i | `10` | <code>int</code> |
| `nOfTargets` | i | `3` | <code>int</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.problem.synthetic.MultiModalIntOneMax">MultiModalIntOneMax</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.SyntheticProblems.multiModalIntOneMax()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.problem.synthetic.multiObjectiveIntOneMax()`

`ea.p.s.multiObjectiveIntOneMax(name; p; upperBound)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `name` | s | interpolate `moIOneMax-{p}` | <code><abbr title="java.lang.String">String</abbr></code> |
| `p` | i | `100` | <code>int</code> |
| `upperBound` | i | `3` | <code>int</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.problem.synthetic.MultiObjectiveIntOneMax">MultiObjectiveIntOneMax</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.SyntheticProblems.multiObjectiveIntOneMax()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.problem.synthetic.oneMax()`

`ea.p.s.oneMax(name; p)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `name` | s | interpolate `oneMax-{p}` | <code><abbr title="java.lang.String">String</abbr></code> |
| `p` | i | `100` | <code>int</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.problem.synthetic.OneMax">OneMax</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.SyntheticProblems.oneMax()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.problem.synthetic.pointAiming()`

`ea.p.s.pointAiming(name; p; target)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `name` | s | interpolate `pointAiming-{p}` | <code><abbr title="java.lang.String">String</abbr></code> |
| `p` | i | `100` | <code>int</code> |
| `target` | d | `1.0` | <code>double</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.problem.synthetic.numerical.PointsAiming">PointsAiming</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.SyntheticProblems.pointAiming()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.problem.synthetic.pointsAiming2D()`

`ea.p.s.pointsAiming2D(name; targetY; targetXs)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `name` | s | interpolate `pointsAiming` | <code><abbr title="java.lang.String">String</abbr></code> |
| `targetY` | d | `2.0` | <code>double</code> |
| `targetXs` | d[] | `[-2.0, 2.0]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.Double">Double</abbr>&gt;</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.problem.synthetic.numerical.PointsAiming">PointsAiming</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.SyntheticProblems.pointsAiming2D()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.problem.synthetic.rastrigin()`

`ea.p.s.rastrigin(name; p)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `name` | s | interpolate `rastrigin-{p}` | <code><abbr title="java.lang.String">String</abbr></code> |
| `p` | i | `100` | <code>int</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.problem.synthetic.numerical.Rastrigin">Rastrigin</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.SyntheticProblems.rastrigin()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.problem.synthetic.rosenbrock()`

`ea.p.s.rosenbrock(name; p)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `name` | s | interpolate `rosenbrock-{p}` | <code><abbr title="java.lang.String">String</abbr></code> |
| `p` | i | `100` | <code>int</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.problem.synthetic.numerical.Rosenbrock">Rosenbrock</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.SyntheticProblems.rosenbrock()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.problem.synthetic.sphere()`

`ea.p.s.sphere(name; p)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `name` | s | interpolate `sphere-{p}` | <code><abbr title="java.lang.String">String</abbr></code> |
| `p` | i | `100` | <code>int</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.problem.synthetic.numerical.Sphere">Sphere</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.SyntheticProblems.sphere()` by jgea-experimenter:2.7.1-SNAPSHOT

## Package `ea.problem.univariateRegression`

Aliases: `ea.p.univariateRegression`, `ea.p.ur`, `ea.problem.univariateRegression`, `ea.problem.ur`

### Builder `ea.problem.univariateRegression.bundled()`

`ea.p.ur.bundled(provider; metrics; nFolds; testFold; name; xScaling; yScaling)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `provider` | npm | `ea.provider.num.fromBundled(yScaling = none; xScaling = none; name = null)` | <code><abbr title="io.github.ericmedvet.jgea.core.util.IndexedProvider">IndexedProvider</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.problem.ExampleBasedProblem$Example">ExampleBasedProblem$Example</abbr>&lt;<abbr title="java.util.Map">Map</abbr>&lt;<abbr title="java.lang.String">String</abbr>, <abbr title="java.lang.Double">Double</abbr>&gt;, <abbr title="java.util.Map">Map</abbr>&lt;<abbr title="java.lang.String">String</abbr>, <abbr title="java.lang.Double">Double</abbr>&gt;&gt;&gt;</code> |
| `metrics` | e[] | `[mse]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.problem.regression.univariate.UnivariateRegressionProblem$Metric">UnivariateRegressionProblem$Metric</abbr>&gt;</code> |
| `nFolds` | i | `10` | <code>int</code> |
| `testFold` | i | `0` | <code>int</code> |
| `name` | s | `` | <code><abbr title="java.lang.String">String</abbr></code> |
| `xScaling` | s | `none` | <code><abbr title="java.lang.String">String</abbr></code> |
| `yScaling` | s | `none` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.problem.regression.univariate.UnivariateRegressionProblem">UnivariateRegressionProblem</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.UnivariateRegressionProblems.fromData()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.problem.univariateRegression.fromData()`

`ea.p.ur.fromData(provider; metrics; nFolds; testFold)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `provider` | npm |  | <code><abbr title="io.github.ericmedvet.jgea.core.util.IndexedProvider">IndexedProvider</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.problem.ExampleBasedProblem$Example">ExampleBasedProblem$Example</abbr>&lt;<abbr title="java.util.Map">Map</abbr>&lt;<abbr title="java.lang.String">String</abbr>, <abbr title="java.lang.Double">Double</abbr>&gt;, <abbr title="java.util.Map">Map</abbr>&lt;<abbr title="java.lang.String">String</abbr>, <abbr title="java.lang.Double">Double</abbr>&gt;&gt;&gt;</code> |
| `metrics` | e[] | `[mse]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.problem.regression.univariate.UnivariateRegressionProblem$Metric">UnivariateRegressionProblem$Metric</abbr>&gt;</code> |
| `nFolds` | i | `10` | <code>int</code> |
| `testFold` | i | `0` | <code>int</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.problem.regression.univariate.UnivariateRegressionProblem">UnivariateRegressionProblem</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.UnivariateRegressionProblems.fromData()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.problem.univariateRegression.synthetic()`

`ea.p.ur.synthetic(name; metrics; seed)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `name` | s |  | <code><abbr title="java.lang.String">String</abbr></code> |
| `metrics` | e[] | `[mse]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.problem.regression.univariate.UnivariateRegressionProblem$Metric">UnivariateRegressionProblem$Metric</abbr>&gt;</code> |
| `seed` | i | `1` | <code>int</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.problem.regression.univariate.synthetic.SyntheticURProblem">SyntheticURProblem</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.UnivariateRegressionProblems.synthetic()` by jgea-experimenter:2.7.1-SNAPSHOT

## Package `ea.provider.numerical`

Aliases: `ea.provider.num`, `ea.provider.numerical`

### Builder `ea.provider.numerical.empty()`

`ea.provider.num.empty(xVars; yVars)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `xVars` | s[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.String">String</abbr>&gt;</code> |
| `yVars` | s[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.String">String</abbr>&gt;</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.problem.regression.NumericalDataset">NumericalDataset</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.NumericalIndexedProviders.empty()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.provider.numerical.fromBundled()`

`ea.provider.num.fromBundled(name; xScaling; yScaling; limit)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `name` | s |  | <code><abbr title="java.lang.String">String</abbr></code> |
| `xScaling` | e | `NONE` | <code><abbr title="io.github.ericmedvet.jgea.problem.regression.NumericalDataset$Scaling">NumericalDataset$Scaling</abbr></code> |
| `yScaling` | e | `NONE` | <code><abbr title="io.github.ericmedvet.jgea.problem.regression.NumericalDataset$Scaling">NumericalDataset$Scaling</abbr></code> |
| `limit` | i | `2147483647` | <code>int</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.problem.regression.NumericalDataset">NumericalDataset</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.NumericalIndexedProviders.fromBundled()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.provider.numerical.fromFile()`

`ea.provider.num.fromFile(filePath; xVarNamePattern; yVarNamePattern; limit)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `filePath` | s |  | <code><abbr title="java.lang.String">String</abbr></code> |
| `xVarNamePattern` | s | `x.*` | <code><abbr title="java.lang.String">String</abbr></code> |
| `yVarNamePattern` | s | `y.*` | <code><abbr title="java.lang.String">String</abbr></code> |
| `limit` | i | `2147483647` | <code>int</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.problem.regression.NumericalDataset">NumericalDataset</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.NumericalIndexedProviders.fromFile()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.provider.numerical.scaled()`

`ea.provider.num.scaled(of; xScaling; yScaling)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm |  | <code><abbr title="io.github.ericmedvet.jgea.problem.regression.NumericalDataset">NumericalDataset</abbr></code> |
| `xScaling` | e | `NONE` | <code><abbr title="io.github.ericmedvet.jgea.problem.regression.NumericalDataset$Scaling">NumericalDataset$Scaling</abbr></code> |
| `yScaling` | e | `NONE` | <code><abbr title="io.github.ericmedvet.jgea.problem.regression.NumericalDataset$Scaling">NumericalDataset$Scaling</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.problem.regression.NumericalDataset">NumericalDataset</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.NumericalIndexedProviders.scaled()` by jgea-experimenter:2.7.1-SNAPSHOT

## Package `ea.representation`

Aliases: `ea.r`, `ea.representation`

### Builder `ea.representation.bitString()`

`ea.r.bitString(pMutRate)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `pMutRate` | d | `1.0` | <code>double</code> |

Produces <code><abbr title="java.util.function.Function">Function</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.representation.sequence.bit.BitString">BitString</abbr>, <abbr title="io.github.ericmedvet.jgea.experimenter.Representation">Representation</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.representation.sequence.bit.BitString">BitString</abbr>&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Representations.bitString()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.representation.cfgTree()`

`ea.r.cfgTree(grammar; minTreeH; maxTreeH)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `grammar` | npm |  | <code><abbr title="io.github.ericmedvet.jgea.core.representation.grammar.string.StringGrammar">StringGrammar</abbr>&lt;N&gt;</code> |
| `minTreeH` | i | `4` | <code>int</code> |
| `maxTreeH` | i | `16` | <code>int</code> |

Produces <code><abbr title="java.util.function.Function">Function</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.representation.tree.Tree">Tree</abbr>&lt;N&gt;, <abbr title="io.github.ericmedvet.jgea.experimenter.Representation">Representation</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.representation.tree.Tree">Tree</abbr>&lt;N&gt;&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Representations.cfgTree()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.representation.doubleString()`

`ea.r.doubleString(initialMinV; initialMaxV; sigmaMut)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `initialMinV` | d | `-1.0` | <code>double</code> |
| `initialMaxV` | d | `1.0` | <code>double</code> |
| `sigmaMut` | d | `0.35` | <code>double</code> |

Produces <code><abbr title="java.util.function.Function">Function</abbr>&lt;<abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.Double">Double</abbr>&gt;, <abbr title="io.github.ericmedvet.jgea.experimenter.Representation">Representation</abbr>&lt;<abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.Double">Double</abbr>&gt;&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Representations.doubleString()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.representation.intString()`

`ea.r.intString(pMutRate)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `pMutRate` | d | `1.0` | <code>double</code> |

Produces <code><abbr title="java.util.function.Function">Function</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.representation.sequence.integer.IntString">IntString</abbr>, <abbr title="io.github.ericmedvet.jgea.experimenter.Representation">Representation</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.representation.sequence.integer.IntString">IntString</abbr>&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Representations.intString()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.representation.multiSRTree()`

`ea.r.multiSRTree(constants; operators; minTreeH; maxTreeH)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `constants` | d[] | `[0.1, 1.0, 10.0]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.Double">Double</abbr>&gt;</code> |
| `operators` | e[] | `[+, -, *, p÷, plog]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.representation.tree.numeric.Element$Operator">Element$Operator</abbr>&gt;</code> |
| `minTreeH` | i | `4` | <code>int</code> |
| `maxTreeH` | i | `10` | <code>int</code> |

Produces <code><abbr title="java.util.function.Function">Function</abbr>&lt;<abbr title="java.util.List">List</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.representation.tree.Tree">Tree</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.representation.tree.numeric.Element">Element</abbr>&gt;&gt;, <abbr title="io.github.ericmedvet.jgea.experimenter.Representation">Representation</abbr>&lt;<abbr title="java.util.List">List</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.representation.tree.Tree">Tree</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.representation.tree.numeric.Element">Element</abbr>&gt;&gt;&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Representations.multiSRTree()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.representation.pair()`

`ea.r.pair(first; second)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `first` | npm |  | <code><abbr title="java.util.function.Function">Function</abbr>&lt;G1, <abbr title="io.github.ericmedvet.jgea.experimenter.Representation">Representation</abbr>&lt;G1&gt;&gt;</code> |
| `second` | npm |  | <code><abbr title="java.util.function.Function">Function</abbr>&lt;G2, <abbr title="io.github.ericmedvet.jgea.experimenter.Representation">Representation</abbr>&lt;G2&gt;&gt;</code> |

Produces <code><abbr title="java.util.function.Function">Function</abbr>&lt;<abbr title="io.github.ericmedvet.jnb.datastructure.Pair">Pair</abbr>&lt;G1, G2&gt;, <abbr title="io.github.ericmedvet.jgea.experimenter.Representation">Representation</abbr>&lt;<abbr title="io.github.ericmedvet.jnb.datastructure.Pair">Pair</abbr>&lt;G1, G2&gt;&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Representations.pair()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.representation.srTree()`

`ea.r.srTree(constants; operators; minTreeH; maxTreeH)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `constants` | d[] | `[0.1, 1.0, 10.0]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.Double">Double</abbr>&gt;</code> |
| `operators` | e[] | `[+, -, *, p÷, plog]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.representation.tree.numeric.Element$Operator">Element$Operator</abbr>&gt;</code> |
| `minTreeH` | i | `4` | <code>int</code> |
| `maxTreeH` | i | `10` | <code>int</code> |

Produces <code><abbr title="java.util.function.Function">Function</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.representation.tree.Tree">Tree</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.representation.tree.numeric.Element">Element</abbr>&gt;, <abbr title="io.github.ericmedvet.jgea.experimenter.Representation">Representation</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.representation.tree.Tree">Tree</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.representation.tree.numeric.Element">Element</abbr>&gt;&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Representations.srTree()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.representation.ttpn()`

`ea.r.ttpn(maxNOfGates; maxNOfAttempts; subnetSizeRate; gates; forbiddenGates; avoidDeadGates)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `maxNOfGates` | i | `32` | <code>int</code> |
| `maxNOfAttempts` | i | `10` | <code>int</code> |
| `subnetSizeRate` | d | `0.33` | <code>double</code> |
| `gates` | npm[] | `[ea.ttpn.gate.bAnd(), ea.ttpn.gate.bNot(), ea.ttpn.gate.bOr(), ea.ttpn.gate.bXor(), ea.ttpn.gate.concat(), ea.ttpn.gate.equal(), ea.ttpn.gate.iTh(), ea.ttpn.gate.length(), ea.ttpn.gate.noop(), ea.ttpn.gate.pairer(), ea.ttpn.gate.queuer(), ea.ttpn.gate.select(), ea.ttpn.gate.sequencer(), ea.ttpn.gate.sink(), ea.ttpn.gate.splitter(), ea.ttpn.gate.unpairer(), ea.ttpn.gate.iBefore(), ea.ttpn.gate.iPMathOperator(operator = addition), ea.ttpn.gate.iPMathOperator(operator = subtraction), ea.ttpn.gate.iPMathOperator(operator = multiplication), ea.ttpn.gate.iPMathOperator(operator = division), ea.ttpn.gate.iRange(), ea.ttpn.gate.iSMult(), ea.ttpn.gate.iSPMult(), ea.ttpn.gate.iSPSum(), ea.ttpn.gate.iSSum(), ea.ttpn.gate.iToR(), ea.ttpn.gate.rBefore(), ea.ttpn.gate.repeater(), ea.ttpn.gate.rPMathOperator(operator = addition), ea.ttpn.gate.rPMathOperator(operator = subtraction), ea.ttpn.gate.rPMathOperator(operator = multiplication), ea.ttpn.gate.rPMathOperator(operator = division), ea.ttpn.gate.rSMult(), ea.ttpn.gate.rSPMult(), ea.ttpn.gate.rSPSum(), ea.ttpn.gate.rSSum(), ea.ttpn.gate.rToI(), ea.ttpn.gate.sBefore(), ea.ttpn.gate.sConcat(), ea.ttpn.gate.sSplitter(), ea.ttpn.gate.bConst(value = true), ea.ttpn.gate.iConst(value = 0), ea.ttpn.gate.iConst(value = 1), ea.ttpn.gate.iConst(value = 2), ea.ttpn.gate.iConst(value = 5), ea.ttpn.gate.rConst(value = 0), ea.ttpn.gate.rConst(value = 0.1), ea.ttpn.gate.rConst(value = 0.2), ea.ttpn.gate.rConst(value = 0.5), ea.ttpn.gate.sPSequencer()]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.representation.programsynthesis.ttpn.Gate">Gate</abbr>&gt;</code> |
| `forbiddenGates` | npm[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.representation.programsynthesis.ttpn.Gate">Gate</abbr>&gt;</code> |
| `avoidDeadGates` | b | `true` | <code>boolean</code> |

Produces <code><abbr title="java.util.function.Function">Function</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.representation.programsynthesis.ttpn.Network">Network</abbr>, <abbr title="io.github.ericmedvet.jgea.experimenter.Representation">Representation</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.representation.programsynthesis.ttpn.Network">Network</abbr>&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Representations.ttpn()` by jgea-experimenter:2.7.1-SNAPSHOT

## Package `ea.solver`

Aliases: `ea.s`, `ea.solver`

### Builder `ea.solver.biGa()`

`ea.s.biGa(name; representation; mapper; crossoverP; tournamentRate; minNTournament; nPop; nEval; maxUniquenessAttempts; fitnessReducer; additionalIndividualComparators)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `name` | s | `biGa` | <code><abbr title="java.lang.String">String</abbr></code> |
| `representation` | npm |  | <code><abbr title="java.util.function.Function">Function</abbr>&lt;G, <abbr title="io.github.ericmedvet.jgea.experimenter.Representation">Representation</abbr>&lt;G&gt;&gt;</code> |
| `mapper` | npm | `ea.m.identity()` | <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;G, S&gt;</code> |
| `crossoverP` | d | `0.8` | <code>double</code> |
| `tournamentRate` | d | `0.05` | <code>double</code> |
| `minNTournament` | i | `3` | <code>int</code> |
| `nPop` | i | `100` | <code>int</code> |
| `nEval` | i | `1000` | <code>int</code> |
| `maxUniquenessAttempts` | i | `100` | <code>int</code> |
| `fitnessReducer` | npm |  | <code><abbr title="java.util.function.BinaryOperator">BinaryOperator</abbr>&lt;Q&gt;</code> |
| `additionalIndividualComparators` | npm[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.order.PartialComparator">PartialComparator</abbr>&lt;? super <abbr title="io.github.ericmedvet.jgea.core.solver.Individual">Individual</abbr>&lt;G, S, Q&gt;&gt;&gt;</code> |

Produces <code><abbr title="java.util.function.Function">Function</abbr>&lt;S, <abbr title="io.github.ericmedvet.jgea.core.solver.bi.StandardBiEvolver">StandardBiEvolver</abbr>&lt;G, S, Q, O&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Solvers.biGa()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.solver.biMapElites()`

`ea.s.biMapElites(name; representation; mapper; nPop; nEval; descriptors; fitnessReducer; emptyArchive; additionalIndividualComparators)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `name` | s | `biMe` | <code><abbr title="java.lang.String">String</abbr></code> |
| `representation` | npm |  | <code><abbr title="java.util.function.Function">Function</abbr>&lt;G, <abbr title="io.github.ericmedvet.jgea.experimenter.Representation">Representation</abbr>&lt;G&gt;&gt;</code> |
| `mapper` | npm | `ea.m.identity()` | <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;G, S&gt;</code> |
| `nPop` | i | `100` | <code>int</code> |
| `nEval` | i | `1000` | <code>int</code> |
| `descriptors` | npm[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.solver.mapelites.MapElites$Descriptor">MapElites$Descriptor</abbr>&lt;G, S, Q&gt;&gt;</code> |
| `fitnessReducer` | npm |  | <code><abbr title="java.util.function.BinaryOperator">BinaryOperator</abbr>&lt;Q&gt;</code> |
| `emptyArchive` | b | `false` | <code>boolean</code> |
| `additionalIndividualComparators` | npm[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.order.PartialComparator">PartialComparator</abbr>&lt;? super <abbr title="io.github.ericmedvet.jgea.core.solver.mapelites.MEIndividual">MEIndividual</abbr>&lt;G, S, Q&gt;&gt;&gt;</code> |

Produces <code><abbr title="java.util.function.Function">Function</abbr>&lt;S, <abbr title="io.github.ericmedvet.jgea.core.solver.bi.mapelites.MapElitesBiEvolver">MapElitesBiEvolver</abbr>&lt;G, S, Q, O&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Solvers.biMapElites()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.solver.cabea()`

`ea.s.cabea(name; representation; mapper; keepProbability; crossoverP; nTour; nEval; toroidal; mooreRadius; gridSize; substrate; iComparators)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `name` | s | `cabea` | <code><abbr title="java.lang.String">String</abbr></code> |
| `representation` | npm |  | <code><abbr title="java.util.function.Function">Function</abbr>&lt;G, <abbr title="io.github.ericmedvet.jgea.experimenter.Representation">Representation</abbr>&lt;G&gt;&gt;</code> |
| `mapper` | npm | `ea.m.identity()` | <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;G, S&gt;</code> |
| `keepProbability` | d | `0.0` | <code>double</code> |
| `crossoverP` | d | `0.8` | <code>double</code> |
| `nTour` | i | `1` | <code>int</code> |
| `nEval` | i | `1000` | <code>int</code> |
| `toroidal` | b | `true` | <code>boolean</code> |
| `mooreRadius` | i | `1` | <code>int</code> |
| `gridSize` | i | `11` | <code>int</code> |
| `substrate` | e | `EMPTY` | <code><abbr title="io.github.ericmedvet.jgea.core.solver.cabea.SubstrateFiller$Predefined">SubstrateFiller$Predefined</abbr></code> |
| `iComparators` | npm[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.order.PartialComparator">PartialComparator</abbr>&lt;? super <abbr title="io.github.ericmedvet.jgea.core.solver.Individual">Individual</abbr>&lt;G, S, Q&gt;&gt;&gt;</code> |

Produces <code><abbr title="java.util.function.Function">Function</abbr>&lt;S, <abbr title="io.github.ericmedvet.jgea.core.solver.cabea.CellularAutomataBasedSolver">CellularAutomataBasedSolver</abbr>&lt;G, S, Q&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Solvers.cabea()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.solver.cmaEs()`

`ea.s.cmaEs(name; mapper; initialMinV; initialMaxV; nEval)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `name` | s | `cmaEs` | <code><abbr title="java.lang.String">String</abbr></code> |
| `mapper` | npm | `ea.m.identity()` | <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;<abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.Double">Double</abbr>&gt;, S&gt;</code> |
| `initialMinV` | d | `-1.0` | <code>double</code> |
| `initialMaxV` | d | `1.0` | <code>double</code> |
| `nEval` | i | `1000` | <code>int</code> |

Produces <code><abbr title="java.util.function.Function">Function</abbr>&lt;S, <abbr title="io.github.ericmedvet.jgea.core.solver.es.CMAEvolutionaryStrategy">CMAEvolutionaryStrategy</abbr>&lt;S, Q&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Solvers.cmaEs()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.solver.coMapElites()`

`ea.s.coMapElites(name; representation1; representation2; mapper1; mapper2; merger; descriptors1; descriptors2; nEval; populationSize; nOfOffspring; strategy; neighborRadius; maxNOfNeighbors; iComparators)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `name` | s | interpolate `coMe-{strategy}-{neighborRadius}-{maxNOfNeighbors}` | <code><abbr title="java.lang.String">String</abbr></code> |
| `representation1` | npm |  | <code><abbr title="java.util.function.Function">Function</abbr>&lt;G1, <abbr title="io.github.ericmedvet.jgea.experimenter.Representation">Representation</abbr>&lt;G1&gt;&gt;</code> |
| `representation2` | npm |  | <code><abbr title="java.util.function.Function">Function</abbr>&lt;G2, <abbr title="io.github.ericmedvet.jgea.experimenter.Representation">Representation</abbr>&lt;G2&gt;&gt;</code> |
| `mapper1` | npm | `ea.m.identity()` | <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;G1, S1&gt;</code> |
| `mapper2` | npm | `ea.m.identity()` | <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;G2, S2&gt;</code> |
| `merger` | npm |  | <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;<abbr title="io.github.ericmedvet.jnb.datastructure.Pair">Pair</abbr>&lt;S1, S2&gt;, S&gt;</code> |
| `descriptors1` | npm[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.solver.mapelites.MapElites$Descriptor">MapElites$Descriptor</abbr>&lt;G1, S1, Q&gt;&gt;</code> |
| `descriptors2` | npm[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.solver.mapelites.MapElites$Descriptor">MapElites$Descriptor</abbr>&lt;G2, S2, Q&gt;&gt;</code> |
| `nEval` | i | `1000` | <code>int</code> |
| `populationSize` | i | `100` | <code>int</code> |
| `nOfOffspring` | i | `50` | <code>int</code> |
| `strategy` | e | `IDENTITY` | <code><abbr title="io.github.ericmedvet.jgea.core.solver.mapelites.strategy.CoMEStrategy$Prepared">CoMEStrategy$Prepared</abbr></code> |
| `neighborRadius` | d | `2.0` | <code>double</code> |
| `maxNOfNeighbors` | i | `2` | <code>int</code> |
| `iComparators` | npm[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.order.PartialComparator">PartialComparator</abbr>&lt;? super <abbr title="io.github.ericmedvet.jgea.core.solver.mapelites.CoMEIndividual">CoMEIndividual</abbr>&lt;G1, G2, S1, S2, S, Q&gt;&gt;&gt;</code> |

Produces <code><abbr title="java.util.function.Function">Function</abbr>&lt;S, <abbr title="io.github.ericmedvet.jgea.core.solver.mapelites.CoMapElites">CoMapElites</abbr>&lt;G1, G2, S1, S2, S, Q&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Solvers.coMapElites()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.solver.differentialEvolution()`

`ea.s.differentialEvolution(name; mapper; initialMinV; initialMaxV; populationSize; nEval; differentialWeight; crossoverP; remap)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `name` | s | `de` | <code><abbr title="java.lang.String">String</abbr></code> |
| `mapper` | npm | `ea.m.identity()` | <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;<abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.Double">Double</abbr>&gt;, S&gt;</code> |
| `initialMinV` | d | `-1.0` | <code>double</code> |
| `initialMaxV` | d | `1.0` | <code>double</code> |
| `populationSize` | i | `15` | <code>int</code> |
| `nEval` | i | `1000` | <code>int</code> |
| `differentialWeight` | d | `0.5` | <code>double</code> |
| `crossoverP` | d | `0.8` | <code>double</code> |
| `remap` | b | `false` | <code>boolean</code> |

Produces <code><abbr title="java.util.function.Function">Function</abbr>&lt;S, <abbr title="io.github.ericmedvet.jgea.core.solver.DifferentialEvolution">DifferentialEvolution</abbr>&lt;S, Q&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Solvers.differentialEvolution()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.solver.ga()`

`ea.s.ga(name; representation; mapper; crossoverP; tournamentRate; minNTournament; nPop; nEval; maxUniquenessAttempts; remap; iComparators)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `name` | s | `ga` | <code><abbr title="java.lang.String">String</abbr></code> |
| `representation` | npm |  | <code><abbr title="java.util.function.Function">Function</abbr>&lt;G, <abbr title="io.github.ericmedvet.jgea.experimenter.Representation">Representation</abbr>&lt;G&gt;&gt;</code> |
| `mapper` | npm | `ea.m.identity()` | <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;G, S&gt;</code> |
| `crossoverP` | d | `0.8` | <code>double</code> |
| `tournamentRate` | d | `0.05` | <code>double</code> |
| `minNTournament` | i | `3` | <code>int</code> |
| `nPop` | i | `100` | <code>int</code> |
| `nEval` | i | `1000` | <code>int</code> |
| `maxUniquenessAttempts` | i | `100` | <code>int</code> |
| `remap` | b | `false` | <code>boolean</code> |
| `iComparators` | npm[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.order.PartialComparator">PartialComparator</abbr>&lt;? super <abbr title="io.github.ericmedvet.jgea.core.solver.Individual">Individual</abbr>&lt;G, S, Q&gt;&gt;&gt;</code> |

Produces <code><abbr title="java.util.function.Function">Function</abbr>&lt;S, <abbr title="io.github.ericmedvet.jgea.core.solver.StandardEvolver">StandardEvolver</abbr>&lt;G, S, Q&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Solvers.ga()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.solver.maMapElites2()`

`ea.s.maMapElites2(name; representation; mapper; nPop; nEval; descriptors1; descriptors2; iComparators)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `name` | s | `maMe2` | <code><abbr title="java.lang.String">String</abbr></code> |
| `representation` | npm |  | <code><abbr title="java.util.function.Function">Function</abbr>&lt;G, <abbr title="io.github.ericmedvet.jgea.experimenter.Representation">Representation</abbr>&lt;G&gt;&gt;</code> |
| `mapper` | npm | `ea.m.identity()` | <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;G, S&gt;</code> |
| `nPop` | i | `100` | <code>int</code> |
| `nEval` | i | `1000` | <code>int</code> |
| `descriptors1` | npm[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.solver.mapelites.MapElites$Descriptor">MapElites$Descriptor</abbr>&lt;G, S, Q&gt;&gt;</code> |
| `descriptors2` | npm[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.solver.mapelites.MapElites$Descriptor">MapElites$Descriptor</abbr>&lt;G, S, Q&gt;&gt;</code> |
| `iComparators` | npm[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.order.PartialComparator">PartialComparator</abbr>&lt;? super <abbr title="io.github.ericmedvet.jgea.core.solver.Individual">Individual</abbr>&lt;G, S, Q&gt;&gt;&gt;</code> |

Produces <code><abbr title="java.util.function.Function">Function</abbr>&lt;S, <abbr title="io.github.ericmedvet.jgea.core.solver.mapelites.MultiArchiveMapElites">MultiArchiveMapElites</abbr>&lt;G, S, Q&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Solvers.maMapElites2()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.solver.mapElites()`

`ea.s.mapElites(name; representation; mapper; nPop; nEval; descriptors; iComparators)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `name` | s | `me` | <code><abbr title="java.lang.String">String</abbr></code> |
| `representation` | npm |  | <code><abbr title="java.util.function.Function">Function</abbr>&lt;G, <abbr title="io.github.ericmedvet.jgea.experimenter.Representation">Representation</abbr>&lt;G&gt;&gt;</code> |
| `mapper` | npm | `ea.m.identity()` | <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;G, S&gt;</code> |
| `nPop` | i | `100` | <code>int</code> |
| `nEval` | i | `1000` | <code>int</code> |
| `descriptors` | npm[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.solver.mapelites.MapElites$Descriptor">MapElites$Descriptor</abbr>&lt;G, S, Q&gt;&gt;</code> |
| `iComparators` | npm[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.order.PartialComparator">PartialComparator</abbr>&lt;? super <abbr title="io.github.ericmedvet.jgea.core.solver.mapelites.MEIndividual">MEIndividual</abbr>&lt;G, S, Q&gt;&gt;&gt;</code> |

Produces <code><abbr title="java.util.function.Function">Function</abbr>&lt;S, <abbr title="io.github.ericmedvet.jgea.core.solver.mapelites.MapElites">MapElites</abbr>&lt;G, S, Q&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Solvers.mapElites()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.solver.nsga2()`

`ea.s.nsga2(name; representation; mapper; crossoverP; nPop; nEval; maxUniquenessAttempts; remap; iComparators)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `name` | s | `nsga2` | <code><abbr title="java.lang.String">String</abbr></code> |
| `representation` | npm |  | <code><abbr title="java.util.function.Function">Function</abbr>&lt;G, <abbr title="io.github.ericmedvet.jgea.experimenter.Representation">Representation</abbr>&lt;G&gt;&gt;</code> |
| `mapper` | npm | `ea.m.identity()` | <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;G, S&gt;</code> |
| `crossoverP` | d | `0.8` | <code>double</code> |
| `nPop` | i | `100` | <code>int</code> |
| `nEval` | i | `1000` | <code>int</code> |
| `maxUniquenessAttempts` | i | `100` | <code>int</code> |
| `remap` | b | `false` | <code>boolean</code> |
| `iComparators` | npm[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.order.PartialComparator">PartialComparator</abbr>&lt;? super <abbr title="io.github.ericmedvet.jgea.core.solver.Individual">Individual</abbr>&lt;G, S, Q&gt;&gt;&gt;</code> |

Produces <code><abbr title="java.util.function.Function">Function</abbr>&lt;S, <abbr title="io.github.ericmedvet.jgea.core.solver.NsgaII">NsgaII</abbr>&lt;G, S, Q&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Solvers.nsga2()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.solver.oGraphea()`

`ea.s.oGraphea(name; mapper; minConst; maxConst; nConst; operators; nPop; nEval; arcAdditionRate; arcRemovalRate; nodeAdditionRate; nPop; rankBase; remap; iComparators)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `name` | s | `oGraphea` | <code><abbr title="java.lang.String">String</abbr></code> |
| `mapper` | npm | `ea.m.identity()` | <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.representation.graph.Graph">Graph</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.representation.graph.Node">Node</abbr>, <abbr title="io.github.ericmedvet.jgea.core.representation.graph.numeric.operatorgraph.OperatorGraph$NonValuedArc">OperatorGraph$NonValuedArc</abbr>&gt;, S&gt;</code> |
| `minConst` | d | `0.0` | <code>double</code> |
| `maxConst` | d | `5.0` | <code>double</code> |
| `nConst` | i | `10` | <code>int</code> |
| `operators` | e[] | `[+, -, *, p/, plog]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.representation.graph.numeric.operatorgraph.BaseOperator">BaseOperator</abbr>&gt;</code> |
| `nPop` | i | `100` | <code>int</code> |
| `nEval` | i | `1000` | <code>int</code> |
| `arcAdditionRate` | d | `3.0` | <code>double</code> |
| `arcRemovalRate` | d | `0.1` | <code>double</code> |
| `nodeAdditionRate` | d | `1.0` | <code>double</code> |
| `nPop` | i | `5` | <code>int</code> |
| `rankBase` | d | `0.75` | <code>double</code> |
| `remap` | b | `false` | <code>boolean</code> |
| `iComparators` | npm[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.order.PartialComparator">PartialComparator</abbr>&lt;? super <abbr title="io.github.ericmedvet.jgea.core.solver.Individual">Individual</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.representation.graph.Graph">Graph</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.representation.graph.Node">Node</abbr>, <abbr title="io.github.ericmedvet.jgea.core.representation.graph.numeric.operatorgraph.OperatorGraph$NonValuedArc">OperatorGraph$NonValuedArc</abbr>&gt;, S, Q&gt;&gt;&gt;</code> |

Produces <code><abbr title="java.util.function.Function">Function</abbr>&lt;S, <abbr title="io.github.ericmedvet.jgea.core.solver.speciation.SpeciatedEvolver">SpeciatedEvolver</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.representation.graph.Graph">Graph</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.representation.graph.Node">Node</abbr>, <abbr title="io.github.ericmedvet.jgea.core.representation.graph.numeric.operatorgraph.OperatorGraph$NonValuedArc">OperatorGraph$NonValuedArc</abbr>&gt;, S, Q&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Solvers.oGraphea()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.solver.openAiEs()`

`ea.s.openAiEs(name; mapper; initialMinV; initialMaxV; sigma; batchSize; stepSize; beta1; beta2; epsilon; nEval)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `name` | s | `openAiEs` | <code><abbr title="java.lang.String">String</abbr></code> |
| `mapper` | npm | `ea.m.identity()` | <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;<abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.Double">Double</abbr>&gt;, S&gt;</code> |
| `initialMinV` | d | `-1.0` | <code>double</code> |
| `initialMaxV` | d | `1.0` | <code>double</code> |
| `sigma` | d | `0.02` | <code>double</code> |
| `batchSize` | i | `30` | <code>int</code> |
| `stepSize` | d | `0.02` | <code>double</code> |
| `beta1` | d | `0.9` | <code>double</code> |
| `beta2` | d | `0.999` | <code>double</code> |
| `epsilon` | d | `1.0E-8` | <code>double</code> |
| `nEval` | i | `1000` | <code>int</code> |

Produces <code><abbr title="java.util.function.Function">Function</abbr>&lt;S, <abbr title="io.github.ericmedvet.jgea.core.solver.es.OpenAIEvolutionaryStrategy">OpenAIEvolutionaryStrategy</abbr>&lt;S, Q&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Solvers.openAiEs()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.solver.pso()`

`ea.s.pso(name; mapper; initialMinV; initialMaxV; nEval; nPop; w; phiParticle; phiGlobal)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `name` | s | `pso` | <code><abbr title="java.lang.String">String</abbr></code> |
| `mapper` | npm | `ea.m.identity()` | <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;<abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.Double">Double</abbr>&gt;, S&gt;</code> |
| `initialMinV` | d | `-1.0` | <code>double</code> |
| `initialMaxV` | d | `1.0` | <code>double</code> |
| `nEval` | i | `1000` | <code>int</code> |
| `nPop` | i | `100` | <code>int</code> |
| `w` | d | `0.8` | <code>double</code> |
| `phiParticle` | d | `1.5` | <code>double</code> |
| `phiGlobal` | d | `1.5` | <code>double</code> |

Produces <code><abbr title="java.util.function.Function">Function</abbr>&lt;S, <abbr title="io.github.ericmedvet.jgea.core.solver.pso.ParticleSwarmOptimization">ParticleSwarmOptimization</abbr>&lt;S, Q&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Solvers.pso()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.solver.randomSearch()`

`ea.s.randomSearch(name; representation; mapper; nEval; iComparators)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `name` | s | `rs` | <code><abbr title="java.lang.String">String</abbr></code> |
| `representation` | npm |  | <code><abbr title="java.util.function.Function">Function</abbr>&lt;G, <abbr title="io.github.ericmedvet.jgea.experimenter.Representation">Representation</abbr>&lt;G&gt;&gt;</code> |
| `mapper` | npm | `ea.m.identity()` | <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;G, S&gt;</code> |
| `nEval` | i | `1000` | <code>int</code> |
| `iComparators` | npm[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.order.PartialComparator">PartialComparator</abbr>&lt;? super <abbr title="io.github.ericmedvet.jgea.core.solver.Individual">Individual</abbr>&lt;G, S, Q&gt;&gt;&gt;</code> |

Produces <code><abbr title="java.util.function.Function">Function</abbr>&lt;S, <abbr title="io.github.ericmedvet.jgea.core.solver.RandomSearch">RandomSearch</abbr>&lt;G, S, Q&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Solvers.randomSearch()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.solver.randomWalk()`

`ea.s.randomWalk(name; representation; mapper; nEval; iComparators)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `name` | s | `rw` | <code><abbr title="java.lang.String">String</abbr></code> |
| `representation` | npm |  | <code><abbr title="java.util.function.Function">Function</abbr>&lt;G, <abbr title="io.github.ericmedvet.jgea.experimenter.Representation">Representation</abbr>&lt;G&gt;&gt;</code> |
| `mapper` | npm | `ea.m.identity()` | <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;G, S&gt;</code> |
| `nEval` | i | `1000` | <code>int</code> |
| `iComparators` | npm[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.order.PartialComparator">PartialComparator</abbr>&lt;? super <abbr title="io.github.ericmedvet.jgea.core.solver.Individual">Individual</abbr>&lt;G, S, Q&gt;&gt;&gt;</code> |

Produces <code><abbr title="java.util.function.Function">Function</abbr>&lt;S, <abbr title="io.github.ericmedvet.jgea.core.solver.RandomWalk">RandomWalk</abbr>&lt;G, S, Q&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Solvers.randomWalk()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.solver.simpleEs()`

`ea.s.simpleEs(name; mapper; initialMinV; initialMaxV; sigma; parentsRate; nOfElites; nPop; nEval; remap)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `name` | s | `es` | <code><abbr title="java.lang.String">String</abbr></code> |
| `mapper` | npm | `ea.m.identity()` | <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;<abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.Double">Double</abbr>&gt;, S&gt;</code> |
| `initialMinV` | d | `-1.0` | <code>double</code> |
| `initialMaxV` | d | `1.0` | <code>double</code> |
| `sigma` | d | `0.35` | <code>double</code> |
| `parentsRate` | d | `0.33` | <code>double</code> |
| `nOfElites` | i | `1` | <code>int</code> |
| `nPop` | i | `30` | <code>int</code> |
| `nEval` | i | `1000` | <code>int</code> |
| `remap` | b | `false` | <code>boolean</code> |

Produces <code><abbr title="java.util.function.Function">Function</abbr>&lt;S, <abbr title="io.github.ericmedvet.jgea.core.solver.es.SimpleEvolutionaryStrategy">SimpleEvolutionaryStrategy</abbr>&lt;S, Q&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Solvers.simpleEs()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.solver.srGp()`

`ea.s.srGp(name; representation; mapper; crossoverP; tournamentRate; minNTournament; nPop; nEval; maxUniquenessAttempts; remap; iComparators)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `name` | s | `srGp` | <code><abbr title="java.lang.String">String</abbr></code> |
| `representation` | npm | `ea.r.srTree()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;G, <abbr title="io.github.ericmedvet.jgea.experimenter.Representation">Representation</abbr>&lt;G&gt;&gt;</code> |
| `mapper` | npm | `ea.m.srTreeToNurf()` | <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;G, S&gt;</code> |
| `crossoverP` | d | `0.8` | <code>double</code> |
| `tournamentRate` | d | `0.05` | <code>double</code> |
| `minNTournament` | i | `3` | <code>int</code> |
| `nPop` | i | `100` | <code>int</code> |
| `nEval` | i | `1000` | <code>int</code> |
| `maxUniquenessAttempts` | i | `100` | <code>int</code> |
| `remap` | b | `false` | <code>boolean</code> |
| `iComparators` | npm[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.order.PartialComparator">PartialComparator</abbr>&lt;? super <abbr title="io.github.ericmedvet.jgea.core.solver.Individual">Individual</abbr>&lt;G, S, Q&gt;&gt;&gt;</code> |

Produces <code><abbr title="java.util.function.Function">Function</abbr>&lt;S, <abbr title="io.github.ericmedvet.jgea.core.solver.StandardEvolver">StandardEvolver</abbr>&lt;G, S, Q&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Solvers.ga()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.solver.ttpnGp()`

`ea.s.ttpnGp(name; representation; mapper; crossoverP; tournamentRate; minNTournament; nPop; nEval; maxUniquenessAttempts; remap; iComparators)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `name` | s | `ttpnGp` | <code><abbr title="java.lang.String">String</abbr></code> |
| `representation` | npm | `ea.r.ttpn()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;G, <abbr title="io.github.ericmedvet.jgea.experimenter.Representation">Representation</abbr>&lt;G&gt;&gt;</code> |
| `mapper` | npm | `ea.m.ttpnToProgram()` | <code><abbr title="io.github.ericmedvet.jgea.core.InvertibleMapper">InvertibleMapper</abbr>&lt;G, S&gt;</code> |
| `crossoverP` | d | `0.5` | <code>double</code> |
| `tournamentRate` | d | `0.05` | <code>double</code> |
| `minNTournament` | i | `3` | <code>int</code> |
| `nPop` | i | `100` | <code>int</code> |
| `nEval` | i | `1000` | <code>int</code> |
| `maxUniquenessAttempts` | i | `100` | <code>int</code> |
| `remap` | b | `false` | <code>boolean</code> |
| `iComparators` | npm[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.order.PartialComparator">PartialComparator</abbr>&lt;? super <abbr title="io.github.ericmedvet.jgea.core.solver.Individual">Individual</abbr>&lt;G, S, Q&gt;&gt;&gt;</code> |

Produces <code><abbr title="java.util.function.Function">Function</abbr>&lt;S, <abbr title="io.github.ericmedvet.jgea.core.solver.StandardEvolver">StandardEvolver</abbr>&lt;G, S, Q&gt;&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.Solvers.ga()` by jgea-experimenter:2.7.1-SNAPSHOT

## Package `ea.solver.mapelites.descriptor`

Aliases: `ea.s.mapelites.d`, `ea.s.mapelites.descriptor`, `ea.s.me.d`, `ea.s.me.descriptor`, `ea.solver.mapelites.d`, `ea.solver.mapelites.descriptor`, `ea.solver.me.d`, `ea.solver.me.descriptor`

### Builder `ea.solver.mapelites.descriptor.descriptor()`

`ea.s.me.d.descriptor(f; min; max; nOfBins)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `f` | npm |  | <code><abbr title="java.util.function.Function">Function</abbr>&lt;<abbr title="io.github.ericmedvet.jgea.core.solver.Individual">Individual</abbr>&lt;G, S, Q&gt;, <abbr title="java.lang.Number">Number</abbr>&gt;</code> |
| `min` | d | `0.0` | <code>double</code> |
| `max` | d | `1.0` | <code>double</code> |
| `nOfBins` | i | `20` | <code>int</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.core.solver.mapelites.MapElites$Descriptor">MapElites$Descriptor</abbr>&lt;G, S, Q&gt;</code>; built from `io.github.ericmedvet.jgea.experimenter.builders.MapElitesDescriptors.descriptor()` by jgea-experimenter:2.7.1-SNAPSHOT

## Package `ea.ttpn.gate`

### Builder `ea.ttpn.gate.bAnd()`

`ea.ttpn.gate.bAnd()`

Produces <code><abbr title="io.github.ericmedvet.jgea.core.representation.programsynthesis.ttpn.Gate">Gate</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.NetworkGates.bAnd()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.ttpn.gate.bConst()`

`ea.ttpn.gate.bConst(value)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `value` | b | `false` | <code>boolean</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.core.representation.programsynthesis.ttpn.Gate">Gate</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.NetworkGates.bConst()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.ttpn.gate.bNot()`

`ea.ttpn.gate.bNot()`

Produces <code><abbr title="io.github.ericmedvet.jgea.core.representation.programsynthesis.ttpn.Gate">Gate</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.NetworkGates.bNot()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.ttpn.gate.bOr()`

`ea.ttpn.gate.bOr()`

Produces <code><abbr title="io.github.ericmedvet.jgea.core.representation.programsynthesis.ttpn.Gate">Gate</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.NetworkGates.bOr()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.ttpn.gate.bXor()`

`ea.ttpn.gate.bXor()`

Produces <code><abbr title="io.github.ericmedvet.jgea.core.representation.programsynthesis.ttpn.Gate">Gate</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.NetworkGates.bXor()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.ttpn.gate.concat()`

`ea.ttpn.gate.concat()`

Produces <code><abbr title="io.github.ericmedvet.jgea.core.representation.programsynthesis.ttpn.Gate">Gate</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.NetworkGates.concat()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.ttpn.gate.equal()`

`ea.ttpn.gate.equal()`

Produces <code><abbr title="io.github.ericmedvet.jgea.core.representation.programsynthesis.ttpn.Gate">Gate</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.NetworkGates.equal()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.ttpn.gate.iBefore()`

`ea.ttpn.gate.iBefore()`

Produces <code><abbr title="io.github.ericmedvet.jgea.core.representation.programsynthesis.ttpn.Gate">Gate</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.NetworkGates.iBefore()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.ttpn.gate.iConst()`

`ea.ttpn.gate.iConst(value)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `value` | i |  | <code>int</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.core.representation.programsynthesis.ttpn.Gate">Gate</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.NetworkGates.iConst()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.ttpn.gate.iPMathOperator()`

`ea.ttpn.gate.iPMathOperator(operator)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `operator` | e |  | <code><abbr title="io.github.ericmedvet.jgea.core.representation.tree.numeric.Element$Operator">Element$Operator</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.core.representation.programsynthesis.ttpn.Gate">Gate</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.NetworkGates.iPMathOperator()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.ttpn.gate.iRange()`

`ea.ttpn.gate.iRange()`

Produces <code><abbr title="io.github.ericmedvet.jgea.core.representation.programsynthesis.ttpn.Gate">Gate</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.NetworkGates.iRange()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.ttpn.gate.iSMult()`

`ea.ttpn.gate.iSMult()`

Produces <code><abbr title="io.github.ericmedvet.jgea.core.representation.programsynthesis.ttpn.Gate">Gate</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.NetworkGates.iSMult()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.ttpn.gate.iSPMult()`

`ea.ttpn.gate.iSPMult()`

Produces <code><abbr title="io.github.ericmedvet.jgea.core.representation.programsynthesis.ttpn.Gate">Gate</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.NetworkGates.iSPMult()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.ttpn.gate.iSPSum()`

`ea.ttpn.gate.iSPSum()`

Produces <code><abbr title="io.github.ericmedvet.jgea.core.representation.programsynthesis.ttpn.Gate">Gate</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.NetworkGates.iSPSum()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.ttpn.gate.iSSum()`

`ea.ttpn.gate.iSSum()`

Produces <code><abbr title="io.github.ericmedvet.jgea.core.representation.programsynthesis.ttpn.Gate">Gate</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.NetworkGates.iSSum()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.ttpn.gate.iTh()`

`ea.ttpn.gate.iTh()`

Produces <code><abbr title="io.github.ericmedvet.jgea.core.representation.programsynthesis.ttpn.Gate">Gate</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.NetworkGates.iTh()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.ttpn.gate.iToR()`

`ea.ttpn.gate.iToR()`

Produces <code><abbr title="io.github.ericmedvet.jgea.core.representation.programsynthesis.ttpn.Gate">Gate</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.NetworkGates.iToR()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.ttpn.gate.length()`

`ea.ttpn.gate.length()`

Produces <code><abbr title="io.github.ericmedvet.jgea.core.representation.programsynthesis.ttpn.Gate">Gate</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.NetworkGates.length()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.ttpn.gate.noop()`

`ea.ttpn.gate.noop()`

Produces <code><abbr title="io.github.ericmedvet.jgea.core.representation.programsynthesis.ttpn.Gate">Gate</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.NetworkGates.noop()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.ttpn.gate.pairer()`

`ea.ttpn.gate.pairer()`

Produces <code><abbr title="io.github.ericmedvet.jgea.core.representation.programsynthesis.ttpn.Gate">Gate</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.NetworkGates.pairer()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.ttpn.gate.queuer()`

`ea.ttpn.gate.queuer()`

Produces <code><abbr title="io.github.ericmedvet.jgea.core.representation.programsynthesis.ttpn.Gate">Gate</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.NetworkGates.queuer()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.ttpn.gate.rBefore()`

`ea.ttpn.gate.rBefore()`

Produces <code><abbr title="io.github.ericmedvet.jgea.core.representation.programsynthesis.ttpn.Gate">Gate</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.NetworkGates.rBefore()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.ttpn.gate.rConst()`

`ea.ttpn.gate.rConst(value)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `value` | d |  | <code>double</code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.core.representation.programsynthesis.ttpn.Gate">Gate</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.NetworkGates.rConst()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.ttpn.gate.rPMathOperator()`

`ea.ttpn.gate.rPMathOperator(operator)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `operator` | e |  | <code><abbr title="io.github.ericmedvet.jgea.core.representation.tree.numeric.Element$Operator">Element$Operator</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.core.representation.programsynthesis.ttpn.Gate">Gate</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.NetworkGates.rPMathOperator()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.ttpn.gate.rSMult()`

`ea.ttpn.gate.rSMult()`

Produces <code><abbr title="io.github.ericmedvet.jgea.core.representation.programsynthesis.ttpn.Gate">Gate</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.NetworkGates.rSMult()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.ttpn.gate.rSPMult()`

`ea.ttpn.gate.rSPMult()`

Produces <code><abbr title="io.github.ericmedvet.jgea.core.representation.programsynthesis.ttpn.Gate">Gate</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.NetworkGates.rSPMult()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.ttpn.gate.rSPSum()`

`ea.ttpn.gate.rSPSum()`

Produces <code><abbr title="io.github.ericmedvet.jgea.core.representation.programsynthesis.ttpn.Gate">Gate</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.NetworkGates.rSPSum()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.ttpn.gate.rSSum()`

`ea.ttpn.gate.rSSum()`

Produces <code><abbr title="io.github.ericmedvet.jgea.core.representation.programsynthesis.ttpn.Gate">Gate</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.NetworkGates.rSSum()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.ttpn.gate.rToI()`

`ea.ttpn.gate.rToI()`

Produces <code><abbr title="io.github.ericmedvet.jgea.core.representation.programsynthesis.ttpn.Gate">Gate</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.NetworkGates.rToI()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.ttpn.gate.repeater()`

`ea.ttpn.gate.repeater()`

Produces <code><abbr title="io.github.ericmedvet.jgea.core.representation.programsynthesis.ttpn.Gate">Gate</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.NetworkGates.repeater()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.ttpn.gate.sBefore()`

`ea.ttpn.gate.sBefore()`

Produces <code><abbr title="io.github.ericmedvet.jgea.core.representation.programsynthesis.ttpn.Gate">Gate</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.NetworkGates.sBefore()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.ttpn.gate.sConcat()`

`ea.ttpn.gate.sConcat()`

Produces <code><abbr title="io.github.ericmedvet.jgea.core.representation.programsynthesis.ttpn.Gate">Gate</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.NetworkGates.sConcat()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.ttpn.gate.sConst()`

`ea.ttpn.gate.sConst(value)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `value` | s |  | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jgea.core.representation.programsynthesis.ttpn.Gate">Gate</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.NetworkGates.sConst()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.ttpn.gate.sPSequencer()`

`ea.ttpn.gate.sPSequencer()`

Produces <code><abbr title="io.github.ericmedvet.jgea.core.representation.programsynthesis.ttpn.Gate">Gate</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.NetworkGates.sPSequencer()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.ttpn.gate.sSplitter()`

`ea.ttpn.gate.sSplitter()`

Produces <code><abbr title="io.github.ericmedvet.jgea.core.representation.programsynthesis.ttpn.Gate">Gate</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.NetworkGates.sSplitter()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.ttpn.gate.select()`

`ea.ttpn.gate.select()`

Produces <code><abbr title="io.github.ericmedvet.jgea.core.representation.programsynthesis.ttpn.Gate">Gate</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.NetworkGates.select()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.ttpn.gate.sequencer()`

`ea.ttpn.gate.sequencer()`

Produces <code><abbr title="io.github.ericmedvet.jgea.core.representation.programsynthesis.ttpn.Gate">Gate</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.NetworkGates.sequencer()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.ttpn.gate.sink()`

`ea.ttpn.gate.sink()`

Produces <code><abbr title="io.github.ericmedvet.jgea.core.representation.programsynthesis.ttpn.Gate">Gate</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.NetworkGates.sink()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.ttpn.gate.splitter()`

`ea.ttpn.gate.splitter()`

Produces <code><abbr title="io.github.ericmedvet.jgea.core.representation.programsynthesis.ttpn.Gate">Gate</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.NetworkGates.splitter()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `ea.ttpn.gate.unpairer()`

`ea.ttpn.gate.unpairer()`

Produces <code><abbr title="io.github.ericmedvet.jgea.core.representation.programsynthesis.ttpn.Gate">Gate</abbr></code>; built from `io.github.ericmedvet.jgea.experimenter.builders.NetworkGates.unpairer()` by jgea-experimenter:2.7.1-SNAPSHOT

## Package `function`

Aliases: `f`, `function`

### Builder `function.all()`

`f.all(of; fs; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, T&gt;</code> |
| `fs` | npm[] | `[f.identity()]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.util.function.Function">Function</abbr>&lt;T, K&gt;&gt;</code> |
| `format` | s | `%s` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.NamedFunction">NamedFunction</abbr>&lt;X, <abbr title="java.util.List">List</abbr>&lt;K&gt;&gt;</code>; built from `io.github.ericmedvet.jnb.buildable.Functions.all()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `function.any()`

`f.any(of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="java.util.Collection">Collection</abbr>&lt;T&gt;&gt;</code> |
| `format` | s | `%s` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.NamedFunction">NamedFunction</abbr>&lt;X, T&gt;</code>; built from `io.github.ericmedvet.jnb.buildable.Functions.any()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `function.as()`

`f.as(of; name)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, Y&gt;</code> |
| `name` | s |  | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.NamedFunction">NamedFunction</abbr>&lt;X, Y&gt;</code>; built from `io.github.ericmedvet.jnb.buildable.Functions.as()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `function.avg()`

`f.avg(of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="java.util.List">List</abbr>&lt;? extends <abbr title="java.lang.Number">Number</abbr>&gt;&gt;</code> |
| `format` | s | `%.1f` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, <abbr title="java.lang.Double">Double</abbr>&gt;</code>; built from `io.github.ericmedvet.jnb.buildable.Functions.avg()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `function.classSimpleName()`

`f.classSimpleName(of)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="java.lang.Object">Object</abbr>&gt;</code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.NamedFunction">NamedFunction</abbr>&lt;X, <abbr title="java.lang.String">String</abbr>&gt;</code>; built from `io.github.ericmedvet.jnb.buildable.Functions.classSimpleName()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `function.clip()`

`f.clip(of; range; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="java.lang.Double">Double</abbr>&gt;</code> |
| `range` | npm |  | <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code> |
| `format` | s | `%.1f` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, <abbr title="java.lang.Double">Double</abbr>&gt;</code>; built from `io.github.ericmedvet.jnb.buildable.Functions.clip()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `function.composition()`

`f.composition(of; then)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, Z&gt;</code> |
| `then` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;Z, Y&gt;</code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, Y&gt;</code>; built from `io.github.ericmedvet.jnb.buildable.Functions.composition()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `function.distinct()`

`f.distinct(of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="java.util.Collection">Collection</abbr>&lt;T&gt;&gt;</code> |
| `format` | s | `%s` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.NamedFunction">NamedFunction</abbr>&lt;X, <abbr title="java.util.Set">Set</abbr>&lt;T&gt;&gt;</code>; built from `io.github.ericmedvet.jnb.buildable.Functions.distinct()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `function.distinctByKey()`

`f.distinctByKey(of; key; representer; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="java.util.Collection">Collection</abbr>&lt;T&gt;&gt;</code> |
| `key` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;T, K&gt;</code> |
| `representer` | npm | `f.any()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;<abbr title="java.util.Collection">Collection</abbr>&lt;T&gt;, T&gt;</code> |
| `format` | s | `%s` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.NamedFunction">NamedFunction</abbr>&lt;X, <abbr title="java.util.Set">Set</abbr>&lt;T&gt;&gt;</code>; built from `io.github.ericmedvet.jnb.buildable.Functions.distinctByKey()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `function.distinctSortedByKey()`

`f.distinctSortedByKey(of; key; representer; format; sort)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="java.util.Collection">Collection</abbr>&lt;T&gt;&gt;</code> |
| `key` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;T, K&gt;</code> |
| `representer` | npm | `f.first(of = f.sortedBy(by = null))` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;<abbr title="java.util.Collection">Collection</abbr>&lt;T&gt;, T&gt;</code> |
| `format` | s | `%s` | <code><abbr title="java.lang.String">String</abbr></code> |
| `sort` | npm | `` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.NamedFunction">NamedFunction</abbr>&lt;X, <abbr title="java.util.Set">Set</abbr>&lt;T&gt;&gt;</code>; built from `io.github.ericmedvet.jnb.buildable.Functions.distinctByKey()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `function.each()`

`f.each(mapF; of)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `mapF` | npm |  | <code><abbr title="java.util.function.Function">Function</abbr>&lt;T, R&gt;</code> |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="java.util.Collection">Collection</abbr>&lt;T&gt;&gt;</code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.NamedFunction">NamedFunction</abbr>&lt;X, <abbr title="java.util.Collection">Collection</abbr>&lt;R&gt;&gt;</code>; built from `io.github.ericmedvet.jnb.buildable.Functions.each()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `function.filter()`

`f.filter(condition; of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `condition` | npm | `predicate.always()` | <code><abbr title="java.util.function.Predicate">Predicate</abbr>&lt;T&gt;</code> |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="java.util.Collection">Collection</abbr>&lt;T&gt;&gt;</code> |
| `format` | s | `%s` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.NamedFunction">NamedFunction</abbr>&lt;X, <abbr title="java.util.Collection">Collection</abbr>&lt;T&gt;&gt;</code>; built from `io.github.ericmedvet.jnb.buildable.Functions.filter()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `function.first()`

`f.first(n; of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `n` | i | `0` | <code>int</code> |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="java.util.List">List</abbr>&lt;T&gt;&gt;</code> |
| `format` | s | `%s` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.NamedFunction">NamedFunction</abbr>&lt;X, T&gt;</code>; built from `io.github.ericmedvet.jnb.buildable.Functions.nTh()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `function.fromBase64()`

`f.fromBase64(of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="java.lang.String">String</abbr>&gt;</code> |
| `format` | s | `%s` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.NamedFunction">NamedFunction</abbr>&lt;X, <abbr title="java.lang.Object">Object</abbr>&gt;</code>; built from `io.github.ericmedvet.jnb.buildable.Functions.fromBase64()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `function.gridCompactness()`

`f.gridCompactness(predicate; of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `predicate` | npm | `f.nonNull()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;T, <abbr title="java.lang.Boolean">Boolean</abbr>&gt;</code> |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jnb.datastructure.Grid">Grid</abbr>&lt;T&gt;&gt;</code> |
| `format` | s | `%5.3f` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, <abbr title="java.lang.Double">Double</abbr>&gt;</code>; built from `io.github.ericmedvet.jnb.buildable.Functions.gridCompactness()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `function.gridCount()`

`f.gridCount(predicate; of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `predicate` | npm | `f.nonNull()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;T, <abbr title="java.lang.Boolean">Boolean</abbr>&gt;</code> |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jnb.datastructure.Grid">Grid</abbr>&lt;T&gt;&gt;</code> |
| `format` | s | `%2d` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, <abbr title="java.lang.Integer">Integer</abbr>&gt;</code>; built from `io.github.ericmedvet.jnb.buildable.Functions.gridCount()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `function.gridCoverage()`

`f.gridCoverage(predicate; of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `predicate` | npm | `f.nonNull()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;T, <abbr title="java.lang.Boolean">Boolean</abbr>&gt;</code> |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jnb.datastructure.Grid">Grid</abbr>&lt;T&gt;&gt;</code> |
| `format` | s | `%2d` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, <abbr title="java.lang.Double">Double</abbr>&gt;</code>; built from `io.github.ericmedvet.jnb.buildable.Functions.gridCoverage()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `function.gridElongation()`

`f.gridElongation(predicate; of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `predicate` | npm | `f.nonNull()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;T, <abbr title="java.lang.Boolean">Boolean</abbr>&gt;</code> |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jnb.datastructure.Grid">Grid</abbr>&lt;T&gt;&gt;</code> |
| `format` | s | `%5.3f` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, <abbr title="java.lang.Double">Double</abbr>&gt;</code>; built from `io.github.ericmedvet.jnb.buildable.Functions.gridElongation()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `function.gridFitH()`

`f.gridFitH(predicate; of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `predicate` | npm | `f.nonNull()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;T, <abbr title="java.lang.Boolean">Boolean</abbr>&gt;</code> |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jnb.datastructure.Grid">Grid</abbr>&lt;T&gt;&gt;</code> |
| `format` | s | `%2d` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, <abbr title="java.lang.Integer">Integer</abbr>&gt;</code>; built from `io.github.ericmedvet.jnb.buildable.Functions.gridFitH()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `function.gridFitW()`

`f.gridFitW(predicate; of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `predicate` | npm | `f.nonNull()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;T, <abbr title="java.lang.Boolean">Boolean</abbr>&gt;</code> |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jnb.datastructure.Grid">Grid</abbr>&lt;T&gt;&gt;</code> |
| `format` | s | `%2d` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, <abbr title="java.lang.Integer">Integer</abbr>&gt;</code>; built from `io.github.ericmedvet.jnb.buildable.Functions.gridFitW()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `function.gridH()`

`f.gridH(of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jnb.datastructure.Grid">Grid</abbr>&lt;?&gt;&gt;</code> |
| `format` | s | `%2d` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, <abbr title="java.lang.Integer">Integer</abbr>&gt;</code>; built from `io.github.ericmedvet.jnb.buildable.Functions.gridH()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `function.gridW()`

`f.gridW(of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jnb.datastructure.Grid">Grid</abbr>&lt;?&gt;&gt;</code> |
| `format` | s | `%2d` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, <abbr title="java.lang.Integer">Integer</abbr>&gt;</code>; built from `io.github.ericmedvet.jnb.buildable.Functions.gridW()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `function.identity()`

`f.identity()`

Produces <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, X&gt;</code>; built from `io.github.ericmedvet.jnb.buildable.Functions.identity()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `function.last()`

`f.last(n; of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `n` | i | `-1` | <code>int</code> |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="java.util.List">List</abbr>&lt;T&gt;&gt;</code> |
| `format` | s | `%s` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.NamedFunction">NamedFunction</abbr>&lt;X, T&gt;</code>; built from `io.github.ericmedvet.jnb.buildable.Functions.nTh()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `function.mapValue()`

`f.mapValue(key; of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `key` | s |  | <code><abbr title="java.lang.String">String</abbr></code> |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="java.util.Map">Map</abbr>&lt;<abbr title="java.lang.String">String</abbr>, T&gt;&gt;</code> |
| `format` | s | `%s` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, T&gt;</code>; built from `io.github.ericmedvet.jnb.buildable.Functions.mapValue()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `function.mathConst()`

`f.mathConst(v; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `v` | d |  | <code>double</code> |
| `format` | s | `%.1f` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, <abbr title="java.lang.Double">Double</abbr>&gt;</code>; built from `io.github.ericmedvet.jnb.buildable.Functions.mathConst()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `function.mathOp()`

`f.mathOp(of; args; op; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, Y&gt;</code> |
| `args` | npm[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.util.function.Function">Function</abbr>&lt;Y, ? extends <abbr title="java.lang.Number">Number</abbr>&gt;&gt;</code> |
| `op` | e |  | <code><abbr title="io.github.ericmedvet.jnb.core.MathOp">MathOp</abbr></code> |
| `format` | s | `%.1f` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, <abbr title="java.lang.Double">Double</abbr>&gt;</code>; built from `io.github.ericmedvet.jnb.buildable.Functions.mathOp()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `function.max()`

`f.max(of; by; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="java.util.Collection">Collection</abbr>&lt;T&gt;&gt;</code> |
| `by` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;T, C&gt;</code> |
| `format` | s | `%s` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, T&gt;</code>; built from `io.github.ericmedvet.jnb.buildable.Functions.max()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `function.median()`

`f.median(p; by; of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `p` | d | `50` | <code>double</code> |
| `by` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;T, C&gt;</code> |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="java.util.Collection">Collection</abbr>&lt;T&gt;&gt;</code> |
| `format` | s | `%s` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, T&gt;</code>; built from `io.github.ericmedvet.jnb.buildable.Functions.percentile()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `function.min()`

`f.min(of; by; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="java.util.Collection">Collection</abbr>&lt;T&gt;&gt;</code> |
| `by` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;T, C&gt;</code> |
| `format` | s | `%s` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, T&gt;</code>; built from `io.github.ericmedvet.jnb.buildable.Functions.min()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `function.nTh()`

`f.nTh(n; of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `n` | i |  | <code>int</code> |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="java.util.List">List</abbr>&lt;T&gt;&gt;</code> |
| `format` | s | `%s` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.NamedFunction">NamedFunction</abbr>&lt;X, T&gt;</code>; built from `io.github.ericmedvet.jnb.buildable.Functions.nTh()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `function.nThMapValue()`

`f.nThMapValue(n; of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `n` | i |  | <code>int</code> |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="java.util.SequencedMap">SequencedMap</abbr>&lt;<abbr title="java.lang.String">String</abbr>, T&gt;&gt;</code> |
| `format` | s | `%s` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, T&gt;</code>; built from `io.github.ericmedvet.jnb.buildable.Functions.nThMapValue()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `function.nkTh()`

`f.nkTh(n; k; of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `n` | i |  | <code>int</code> |
| `k` | i |  | <code>int</code> |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="java.util.List">List</abbr>&lt;T&gt;&gt;</code> |
| `format` | s | `%s` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, <abbr title="java.util.List">List</abbr>&lt;T&gt;&gt;</code>; built from `io.github.ericmedvet.jnb.buildable.Functions.nkTh()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `function.nonNull()`

`f.nonNull(of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="java.lang.Object">Object</abbr>&gt;</code> |
| `format` | s | `%s` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, <abbr title="java.lang.Boolean">Boolean</abbr>&gt;</code>; built from `io.github.ericmedvet.jnb.buildable.Functions.nonNull()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `function.pairFirst()`

`f.pairFirst(of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jnb.datastructure.Pair">Pair</abbr>&lt;F, S&gt;&gt;</code> |
| `format` | s | `%s` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, F&gt;</code>; built from `io.github.ericmedvet.jnb.buildable.Functions.pairFirst()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `function.pairSecond()`

`f.pairSecond(of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="io.github.ericmedvet.jnb.datastructure.Pair">Pair</abbr>&lt;F, S&gt;&gt;</code> |
| `format` | s | `%s` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, S&gt;</code>; built from `io.github.ericmedvet.jnb.buildable.Functions.pairSecond()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `function.percentile()`

`f.percentile(p; by; of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `p` | d |  | <code>double</code> |
| `by` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;T, C&gt;</code> |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="java.util.Collection">Collection</abbr>&lt;T&gt;&gt;</code> |
| `format` | s | `%s` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, T&gt;</code>; built from `io.github.ericmedvet.jnb.buildable.Functions.percentile()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `function.quantized()`

`f.quantized(q; of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `q` | d |  | <code>double</code> |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="java.lang.Number">Number</abbr>&gt;</code> |
| `format` | s | `%.1f` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, <abbr title="java.lang.Double">Double</abbr>&gt;</code>; built from `io.github.ericmedvet.jnb.buildable.Functions.quantized()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `function.sd()`

`f.sd(of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="java.util.List">List</abbr>&lt;? extends <abbr title="java.lang.Number">Number</abbr>&gt;&gt;</code> |
| `format` | s | `%.1f` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, <abbr title="java.lang.Double">Double</abbr>&gt;</code>; built from `io.github.ericmedvet.jnb.buildable.Functions.sd()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `function.size()`

`f.size(of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="java.util.Collection">Collection</abbr>&lt;?&gt;&gt;</code> |
| `format` | s | `%3d` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, <abbr title="java.lang.Integer">Integer</abbr>&gt;</code>; built from `io.github.ericmedvet.jnb.buildable.Functions.size()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `function.sizeIf()`

`f.sizeIf(of; format; allF; mapF; predicate)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.filter(condition = predicate.always(); of = f.each(of = f.identity(); mapF = f.identity()))` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="java.util.Collection">Collection</abbr>&lt;?&gt;&gt;</code> |
| `format` | s | `%3d` | <code><abbr title="java.lang.String">String</abbr></code> |
| `allF` | npm | `f.identity()` | <code><abbr title="java.lang.String">String</abbr></code> |
| `mapF` | npm | `f.identity()` | <code><abbr title="java.lang.String">String</abbr></code> |
| `predicate` | npm | `predicate.always()` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, <abbr title="java.lang.Integer">Integer</abbr>&gt;</code>; built from `io.github.ericmedvet.jnb.buildable.Functions.size()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `function.sizeIfGt()`

`f.sizeIfGt(of; format; allF; mapF; predicate; t)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.filter(condition = predicate.always(); of = f.each(of = f.identity(); mapF = f.identity()))` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="java.util.Collection">Collection</abbr>&lt;?&gt;&gt;</code> |
| `format` | s | `%3d` | <code><abbr title="java.lang.String">String</abbr></code> |
| `allF` | npm | `f.identity()` | <code><abbr title="java.lang.String">String</abbr></code> |
| `mapF` | npm | `f.identity()` | <code><abbr title="java.lang.String">String</abbr></code> |
| `predicate` | npm | `predicate.gt(t = 0)` | <code><abbr title="java.lang.String">String</abbr></code> |
| `t` | d | `0.0` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, <abbr title="java.lang.Integer">Integer</abbr>&gt;</code>; built from `io.github.ericmedvet.jnb.buildable.Functions.size()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `function.sizeIfLt()`

`f.sizeIfLt(of; format; allF; mapF; predicate; t)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.filter(condition = predicate.always(); of = f.each(of = f.identity(); mapF = f.identity()))` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="java.util.Collection">Collection</abbr>&lt;?&gt;&gt;</code> |
| `format` | s | `%3d` | <code><abbr title="java.lang.String">String</abbr></code> |
| `allF` | npm | `f.identity()` | <code><abbr title="java.lang.String">String</abbr></code> |
| `mapF` | npm | `f.identity()` | <code><abbr title="java.lang.String">String</abbr></code> |
| `predicate` | npm | `predicate.lt(t = 0)` | <code><abbr title="java.lang.String">String</abbr></code> |
| `t` | d | `0.0` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, <abbr title="java.lang.Integer">Integer</abbr>&gt;</code>; built from `io.github.ericmedvet.jnb.buildable.Functions.size()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `function.sortedBy()`

`f.sortedBy(by; of; reversed; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `by` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;T, K&gt;</code> |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="java.util.Collection">Collection</abbr>&lt;T&gt;&gt;</code> |
| `reversed` | b | `false` | <code>boolean</code> |
| `format` | s | `%s` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.NamedFunction">NamedFunction</abbr>&lt;X, <abbr title="java.util.List">List</abbr>&lt;T&gt;&gt;</code>; built from `io.github.ericmedvet.jnb.buildable.Functions.sortedBy()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `function.subList()`

`f.subList(from; to; relative; of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `from` | d |  | <code>double</code> |
| `to` | d |  | <code>double</code> |
| `relative` | b | `true` | <code>boolean</code> |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="java.util.List">List</abbr>&lt;T&gt;&gt;</code> |
| `format` | s | `%s` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, <abbr title="java.util.List">List</abbr>&lt;T&gt;&gt;</code>; built from `io.github.ericmedvet.jnb.buildable.Functions.subList()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `function.toBase64()`

`f.toBase64(of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="java.lang.Object">Object</abbr>&gt;</code> |
| `format` | s | `%s` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.NamedFunction">NamedFunction</abbr>&lt;X, <abbr title="java.lang.String">String</abbr>&gt;</code>; built from `io.github.ericmedvet.jnb.buildable.Functions.toBase64()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `function.toString()`

`f.toString(of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="java.lang.Object">Object</abbr>&gt;</code> |
| `format` | s | `%s` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.NamedFunction">NamedFunction</abbr>&lt;X, <abbr title="java.lang.String">String</abbr>&gt;</code>; built from `io.github.ericmedvet.jnb.buildable.Functions.toString()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `function.uniqueness()`

`f.uniqueness(of; format)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="java.util.Collection">Collection</abbr>&lt;?&gt;&gt;</code> |
| `format` | s | `%5.3f` | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction">FormattedNamedFunction</abbr>&lt;X, <abbr title="java.lang.Double">Double</abbr>&gt;</code>; built from `io.github.ericmedvet.jnb.buildable.Functions.uniqueness()` by jgea-experimenter:2.7.1-SNAPSHOT

## Package `misc`

Aliases: `m`, `misc`

### Builder `misc.defaultRG()`

`m.defaultRG(seed)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `seed` | i | `0` | <code>int</code> |

Produces <code><abbr title="java.util.random.RandomGenerator">RandomGenerator</abbr></code>; built from `io.github.ericmedvet.jnb.buildable.Miscs.defaultRG()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `misc.grid()`

`m.grid(w; h; items)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `w` | i |  | <code>int</code> |
| `h` | i |  | <code>int</code> |
| `items` | npm[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;T&gt;</code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.Grid">Grid</abbr>&lt;T&gt;</code>; built from `io.github.ericmedvet.jnb.buildable.Miscs.grid()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `misc.range()`

`m.range(min; max)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `min` | d |  | <code>double</code> |
| `max` | d |  | <code>double</code> |

Produces <code><abbr title="io.github.ericmedvet.jnb.datastructure.DoubleRange">DoubleRange</abbr></code>; built from `io.github.ericmedvet.jnb.buildable.Miscs.range()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `misc.sharedRG()`

`m.sharedRG(seed)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `seed` | i | `0` | <code>int</code> |

Produces <code><abbr title="java.util.random.RandomGenerator">RandomGenerator</abbr></code>; built from `io.github.ericmedvet.jnb.buildable.Miscs.sharedRG()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `misc.supplier()`

`m.supplier(of)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `of` | npm |  | <code>T</code> |

Produces <code><abbr title="java.util.function.Supplier">Supplier</abbr>&lt;T&gt;</code>; built from `io.github.ericmedvet.jnb.buildable.Miscs.supplier()` by jgea-experimenter:2.7.1-SNAPSHOT

## Package `predicate`

### Builder `predicate.all()`

`predicate.all(conditions)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `conditions` | npm[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.util.function.Predicate">Predicate</abbr>&lt;X&gt;&gt;</code> |

Produces <code><abbr title="java.util.function.Predicate">Predicate</abbr>&lt;X&gt;</code>; built from `io.github.ericmedvet.jnb.buildable.Predicates.all()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `predicate.always()`

`predicate.always()`

Produces <code><abbr title="java.util.function.Predicate">Predicate</abbr>&lt;?&gt;</code>; built from `io.github.ericmedvet.jnb.buildable.Predicates.always()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `predicate.any()`

`predicate.any(conditions)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `conditions` | npm[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.util.function.Predicate">Predicate</abbr>&lt;X&gt;&gt;</code> |

Produces <code><abbr title="java.util.function.Predicate">Predicate</abbr>&lt;X&gt;</code>; built from `io.github.ericmedvet.jnb.buildable.Predicates.any()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `predicate.eq()`

`predicate.eq(f; v)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `f` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, T&gt;</code> |
| `v` | npm |  | <code>T</code> |

Produces <code><abbr title="java.util.function.Predicate">Predicate</abbr>&lt;X&gt;</code>; built from `io.github.ericmedvet.jnb.buildable.Predicates.eq()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `predicate.eqD()`

`predicate.eqD(f; v)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `f` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="java.lang.Double">Double</abbr>&gt;</code> |
| `v` | d |  | <code><abbr title="java.lang.Double">Double</abbr></code> |

Produces <code><abbr title="java.util.function.Predicate">Predicate</abbr>&lt;X&gt;</code>; built from `io.github.ericmedvet.jnb.buildable.Predicates.eqD()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `predicate.eqI()`

`predicate.eqI(f; v)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `f` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="java.lang.Integer">Integer</abbr>&gt;</code> |
| `v` | i |  | <code><abbr title="java.lang.Integer">Integer</abbr></code> |

Produces <code><abbr title="java.util.function.Predicate">Predicate</abbr>&lt;X&gt;</code>; built from `io.github.ericmedvet.jnb.buildable.Predicates.eqI()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `predicate.eqS()`

`predicate.eqS(f; v)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `f` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="java.lang.String">String</abbr>&gt;</code> |
| `v` | s |  | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="java.util.function.Predicate">Predicate</abbr>&lt;X&gt;</code>; built from `io.github.ericmedvet.jnb.buildable.Predicates.eqS()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `predicate.gt()`

`predicate.gt(f; t)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `f` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, ? extends <abbr title="java.lang.Number">Number</abbr>&gt;</code> |
| `t` | d |  | <code>double</code> |

Produces <code><abbr title="java.util.function.Predicate">Predicate</abbr>&lt;X&gt;</code>; built from `io.github.ericmedvet.jnb.buildable.Predicates.gt()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `predicate.gtEq()`

`predicate.gtEq(f; t)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `f` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, ? extends <abbr title="java.lang.Number">Number</abbr>&gt;</code> |
| `t` | d |  | <code>double</code> |

Produces <code><abbr title="java.util.function.Predicate">Predicate</abbr>&lt;X&gt;</code>; built from `io.github.ericmedvet.jnb.buildable.Predicates.gtEq()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `predicate.inD()`

`predicate.inD(f; values)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `f` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="java.lang.Double">Double</abbr>&gt;</code> |
| `values` | d[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.Double">Double</abbr>&gt;</code> |

Produces <code><abbr title="java.util.function.Predicate">Predicate</abbr>&lt;X&gt;</code>; built from `io.github.ericmedvet.jnb.buildable.Predicates.inD()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `predicate.inI()`

`predicate.inI(f; values)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `f` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="java.lang.Integer">Integer</abbr>&gt;</code> |
| `values` | i[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.Integer">Integer</abbr>&gt;</code> |

Produces <code><abbr title="java.util.function.Predicate">Predicate</abbr>&lt;X&gt;</code>; built from `io.github.ericmedvet.jnb.buildable.Predicates.inI()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `predicate.inL()`

`predicate.inL(f; values)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `f` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="java.lang.Long">Long</abbr>&gt;</code> |
| `values` | i[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.Integer">Integer</abbr>&gt;</code> |

Produces <code><abbr title="java.util.function.Predicate">Predicate</abbr>&lt;X&gt;</code>; built from `io.github.ericmedvet.jnb.buildable.Predicates.inL()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `predicate.inS()`

`predicate.inS(f; values)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `f` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="java.lang.String">String</abbr>&gt;</code> |
| `values` | s[] | `[]` | <code><abbr title="java.util.List">List</abbr>&lt;<abbr title="java.lang.String">String</abbr>&gt;</code> |

Produces <code><abbr title="java.util.function.Predicate">Predicate</abbr>&lt;X&gt;</code>; built from `io.github.ericmedvet.jnb.buildable.Predicates.inS()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `predicate.lt()`

`predicate.lt(f; t)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `f` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, ? extends <abbr title="java.lang.Number">Number</abbr>&gt;</code> |
| `t` | d |  | <code>double</code> |

Produces <code><abbr title="java.util.function.Predicate">Predicate</abbr>&lt;X&gt;</code>; built from `io.github.ericmedvet.jnb.buildable.Predicates.lt()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `predicate.ltEq()`

`predicate.ltEq(f; t)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `f` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, ? extends <abbr title="java.lang.Number">Number</abbr>&gt;</code> |
| `t` | d |  | <code>double</code> |

Produces <code><abbr title="java.util.function.Predicate">Predicate</abbr>&lt;X&gt;</code>; built from `io.github.ericmedvet.jnb.buildable.Predicates.ltEq()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `predicate.matches()`

`predicate.matches(f; regex)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `f` | npm | `f.identity()` | <code><abbr title="java.util.function.Function">Function</abbr>&lt;X, <abbr title="java.lang.String">String</abbr>&gt;</code> |
| `regex` | s |  | <code><abbr title="java.lang.String">String</abbr></code> |

Produces <code><abbr title="java.util.function.Predicate">Predicate</abbr>&lt;X&gt;</code>; built from `io.github.ericmedvet.jnb.buildable.Predicates.matches()` by jgea-experimenter:2.7.1-SNAPSHOT

### Builder `predicate.not()`

`predicate.not(condition)`

| Param | Type | Default | Java type |
| --- | --- | --- | --- |
| `condition` | npm |  | <code><abbr title="java.util.function.Predicate">Predicate</abbr>&lt;X&gt;</code> |

Produces <code><abbr title="java.util.function.Predicate">Predicate</abbr>&lt;X&gt;</code>; built from `io.github.ericmedvet.jnb.buildable.Predicates.not()` by jgea-experimenter:2.7.1-SNAPSHOT

