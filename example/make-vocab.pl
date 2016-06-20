#!/usr/bin/perl

use strict;
use warnings;
use utf8;
use Getopt::Long;
use List::Util qw(sum min max shuffle);
binmode STDIN, ":utf8";
binmode STDOUT, ":utf8";
binmode STDERR, ":utf8";

my $LANG = "en";
GetOptions(
"lang=s" => \$LANG,
);

if(@ARGV != 0) {
  print STDERR "Usage: $0\n";
  exit 1;
}

my (@vocab, %map);
$map{"<unk>"}++; push @vocab, "<unk>";
$map{"<s>"}++; push @vocab, "<s>";
while(<STDIN>) {
  chomp;
  for(split(/ /)) {
    if(not $map{$_}++) {
      push @vocab, $_;
    }
  }
}
for(@vocab) {
  print "$_\n";
}
