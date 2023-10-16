module io.github.ericmedvet.jgea.experimenter {
  opens io.github.ericmedvet.jgea.experimenter.builders to io.github.ericmedvet.jnb.core;
  exports io.github.ericmedvet.jgea.experimenter;
  requires io.github.ericmedvet.jnb.core;
  requires io.github.ericmedvet.jgea.core;
  requires io.github.ericmedvet.jgea.tui;
  requires io.github.ericmedvet.jgea.problem;
  requires io.github.ericmedvet.jsdynsym.core;
  requires io.github.ericmedvet.jsdynsym.buildable;
  requires java.logging;
  requires java.desktop;
  requires jdk.management;
  requires com.googlecode.lanterna;
  requires jcommander;
  requires io.github.ericmedvet.jgea.telegram;
  opens io.github.ericmedvet.jgea.experimenter.net to jcommander;
}