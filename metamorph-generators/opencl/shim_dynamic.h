/** Defines necessary to override the behavior of the functions in metamorph_shim.c with treating MetaMorph as an optional plugin.
 * Essentially these defines will attempt to dynamically load Metamorph inside the emulated functions, and if it is found, defer to it.
 * Only if it is not found will the emulation route be used.
 * This file is prepended to metamorph_shim.c automatically if MetaCL is run with --use-metamorph=OPTIONAL
 */
