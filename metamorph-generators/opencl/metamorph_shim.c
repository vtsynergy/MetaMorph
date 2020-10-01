/** \file Emulations of MetaMorph functions critical to MetaCL-generated programs
 * Contains macro references that will NOOP if MetaCL is run with --use-metamorph=DISABLED
 * However, if there is code before this comment block, MetaCL has been run with --use-metamorph=OPTIONAL, and the above macros will attempt to dynamically-load MetaMorph, and only defer to the emulation if the library is not found.
 */
