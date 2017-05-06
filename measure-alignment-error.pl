#!/usr/bin/perl

use strict;
use utf8;
use List::Util qw(sum min max shuffle);
binmode STDIN, ":utf8";
binmode STDOUT, ":utf8";
binmode STDERR, ":utf8";

if(@ARGV != 2) {
    print STDERR "Usage: measure-alignment-error.pl REF TEST\n";
    exit 1;
}

open FILE0, "<:utf8", $ARGV[0] or die "$ARGV[0]: $!\n";
open FILE1, "<:utf8", $ARGV[1] or die "$ARGV[1]: $!\n";

my ($ref, $test, $match_rec, $match_prec);
while((my $f0 = <FILE0>) and (my $f1 = <FILE1>)) {
    chomp $f0; chomp $f1;
    my (%ref_sure, %ref_prob);
    for(split(/ /, $f0)) {
        /(P-|S-|)(\d+-\d+)/ or die "badly formatted alignment\n";
        $ref_prob{$2}++;
        $ref_sure{$2}++ if $1 ne "P-";
    }
    my %test = map { $_ => 1 } split(/ /,$f1);
    $ref += keys %ref_sure;
    $test += keys %test;
    for(keys %ref_sure) {
        $match_rec++ if $test{$_};
    }
    for(keys %test) {
        $match_prec++ if $ref_prob{$_};
    }
}

print "Ref=$ref\tTest=$test\tMatchRef=$match_rec\tMatchTest=$match_prec\n";
my $prec = $match_prec/$test;
my $rec = $match_rec/$ref;
my $fmeas = 2*$prec*$rec/($prec + $rec);
printf "Prec=%.2f\tRec=%.2f\tAER=%.2f\n", $prec*100, $rec*100, (1-$fmeas)*100;
