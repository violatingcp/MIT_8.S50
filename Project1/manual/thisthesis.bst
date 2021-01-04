%%
%% This is file `thisthesis.bst',
%% generated with the docstrip utility.
%%
%% The original source files were:
%%
%% merlin.mbs  (with options: `annote,seq-no,nm-init,ed-au,aunm-semi,nmlm,x1,m1,dt-jnl,yr-par,xmth,yrp-x,tit-qq,qx,atit-u,thtit-a,trnum-it,vnum-sp,volp-sp,jdt-p,jpg-1,jwdpg,jwdvol,num-xser,ser-ed,pub-date,pub-xc,isbn,issn,doi,edpar,bkedcap,edby-par,edbyy,blk-com,in-x,pp,ed,abr,ednx,mth-bare,ord,jabr,etal-it,eprint,nfss,')
%% ----------------------------------------
%% *** thisthesis ***
%% 
%% Copyright 1994-2011 Patrick W Daly
 % ===============================================================
 % IMPORTANT NOTICE:
 % This bibliographic style (bst) file has been generated from one or
 % more master bibliographic style (mbs) files, listed above.
 %
 % This generated file can be redistributed and/or modified under the terms
 % of the LaTeX Project Public License Distributed from CTAN
 % archives in directory macros/latex/base/lppl.txt; either
 % version 1 of the License, or any later version.
 % ===============================================================
 % Name and version information of the main mbs file:
 % \ProvidesFile{merlin.mbs}[2011/11/18 4.33 (PWD, AO, DPC)]
 %   For use with BibTeX version 0.99a or later
 %-------------------------------------------------------------------
 % This bibliography style file is intended for texts in ENGLISH
 % This is a numerical citation style, and as such is standard LaTeX.
 % It requires no extra package to interface to the main text.
 % The form of the \bibitem entries is
 %   \bibitem{key}...
 % Usage of \cite is as follows:
 %   \cite{key} ==>>          [#]
 %   \cite[chap. 2]{key} ==>> [#, chap. 2]
 % where # is a number determined by the ordering in the reference list.
 % The order in the reference list is that by which the works were originally
 %   cited in the text, or that in the database.
 %---------------------------------------------------------------------

ENTRY
  { address
    annote
    archive
    author
    booktitle
    chapter
    collaboration
    doi
    edition
    editor
    eid
    eprint
    howpublished
    institution
    isbn
    issn
    journal
    key
    month
    note
    number
    organization
    pages
    publisher
    school
    series
    title
    type
    volume
    year
    url
  }
  {}
  { label }
INTEGERS { output.state before.all mid.sentence after.sentence after.block }
FUNCTION {init.state.consts}
{ #0 'before.all :=
  #1 'mid.sentence :=
  #2 'after.sentence :=
  #3 'after.block :=
}
STRINGS { s t}
FUNCTION {output.nonnull}
{ 's :=
  output.state mid.sentence =
    { ", " * write$ }
    { output.state after.block =
        { add.period$ write$
          newline$
          "\newblock " write$
        }
        { output.state before.all =
            'write$
            { add.period$ " " * write$ }
          if$
        }
      if$
      mid.sentence 'output.state :=
    }
  if$
  s
}
FUNCTION {output}
{ duplicate$ empty$
    'pop$
    'output.nonnull
  if$
}
FUNCTION {output.check}
{ 't :=
  duplicate$ empty$
    { pop$ "empty " t * " in " * cite$ * warning$ }
    'output.nonnull
  if$
}
FUNCTION {fin.entry}
{ add.period$
  write$
  newline$
  annote missing$
    { "\bibAnnoteFile{" cite$ * "}" * write$ newline$ }
    { "\bibAnnote{" cite$ * "}{" * annote * "}" * write$ newline$
    }
  if$
}

FUNCTION {new.block}
{ output.state before.all =
    'skip$
    { after.block 'output.state := }
  if$
}
FUNCTION {new.sentence}
{ output.state after.block =
    'skip$
    { output.state before.all =
        'skip$
        { after.sentence 'output.state := }
      if$
    }
  if$
}
FUNCTION {add.blank}
{  " " * before.all 'output.state :=
}

FUNCTION {date.block}
{
  add.blank
}

FUNCTION {not}
{   { #0 }
    { #1 }
  if$
}
FUNCTION {and}
{   'skip$
    { pop$ #0 }
  if$
}
FUNCTION {or}
{   { pop$ #1 }
    'skip$
  if$
}
FUNCTION {non.stop}
{ duplicate$
   "}" * add.period$
   #-1 #1 substring$ "." =
}

STRINGS {z}

FUNCTION {remove.dots}
{ 'z :=
   ""
   { z empty$ not }
   { z #1 #2 substring$
     duplicate$ "\." =
       { z #3 global.max$ substring$ 'z :=  * }
       { pop$
         z #1 #1 substring$
         z #2 global.max$ substring$ 'z :=
         duplicate$ "." = 'pop$
           { * }
         if$
       }
     if$
   }
   while$
}
FUNCTION {new.block.checka}
{ empty$
    'skip$
    'new.block
  if$
}
FUNCTION {new.block.checkb}
{ empty$
  swap$ empty$
  and
    'skip$
    'new.block
  if$
}
FUNCTION {new.sentence.checka}
{ empty$
    'skip$
    'new.sentence
  if$
}
FUNCTION {new.sentence.checkb}
{ empty$
  swap$ empty$
  and
    'skip$
    'new.sentence
  if$
}
FUNCTION {field.or.null}
{ duplicate$ empty$
    { pop$ "" }
    'skip$
  if$
}
FUNCTION {emphasize}
{ duplicate$ empty$
    { pop$ "" }
    { "\emph{" swap$ * "}" * }
  if$
}
FUNCTION {tie.or.space.prefix}
{ duplicate$ text.length$ #3 <
    { "~" }
    { " " }
  if$
  swap$
}

FUNCTION {capitalize}
{ "u" change.case$ "t" change.case$ }

FUNCTION {space.word}
{ " " swap$ * " " * }
 % Here are the language-specific definitions for explicit words.
 % Each function has a name bbl.xxx where xxx is the English word.
 % The language selected here is ENGLISH
FUNCTION {bbl.and}
{ "and"}

FUNCTION {bbl.etal}
{ "et~al." }

FUNCTION {bbl.editors}
{ "eds." }

FUNCTION {bbl.editor}
{ "ed." }

FUNCTION {bbl.edby}
{ "edited by" }

FUNCTION {bbl.edition}
{ "ed." }

FUNCTION {bbl.volume}
{ "vol." }

FUNCTION {bbl.of}
{ "of" }

FUNCTION {bbl.number}
{ "no." }

FUNCTION {bbl.nr}
{ "no." }

FUNCTION {bbl.in}
{ "in" }

FUNCTION {bbl.pages}
{ "pp." }

FUNCTION {bbl.page}
{ "p." }

FUNCTION {bbl.chapter}
{ "chap." }

FUNCTION {bbl.techrep}
{ "Tech. Rep." }

FUNCTION {bbl.mthesis}
{ "Master's thesis" }

FUNCTION {bbl.phdthesis}
{ "Ph.D. thesis" }

FUNCTION {bbl.first}
{ "1st" }

FUNCTION {bbl.second}
{ "2nd" }

FUNCTION {bbl.third}
{ "3rd" }

FUNCTION {bbl.fourth}
{ "4th" }

FUNCTION {bbl.fifth}
{ "5th" }

FUNCTION {bbl.st}
{ "st" }

FUNCTION {bbl.nd}
{ "nd" }

FUNCTION {bbl.rd}
{ "rd" }

FUNCTION {bbl.th}
{ "th" }

MACRO {jan} {"Jan."}

MACRO {feb} {"Feb."}

MACRO {mar} {"Mar."}

MACRO {apr} {"Apr."}

MACRO {may} {"May"}

MACRO {jun} {"Jun."}

MACRO {jul} {"Jul."}

MACRO {aug} {"Aug."}

MACRO {sep} {"Sep."}

MACRO {oct} {"Oct."}

MACRO {nov} {"Nov."}

MACRO {dec} {"Dec."}

FUNCTION {eng.ord}
{ duplicate$ "1" swap$ *
  #-2 #1 substring$ "1" =
     { bbl.th * }
     { duplicate$ #-1 #1 substring$
       duplicate$ "1" =
         { pop$ bbl.st * }
         { duplicate$ "2" =
             { pop$ bbl.nd * }
             { "3" =
                 { bbl.rd * }
                 { bbl.th * }
               if$
             }
           if$
          }
       if$
     }
   if$
}

MACRO {acmcs} {"ACM Comput. Surv."}

MACRO {acta} {"Acta Inf."}

MACRO {cacm} {"Commun. ACM"}

MACRO {ibmjrd} {"IBM J. Res. Dev."}

MACRO {ibmsj} {"IBM Syst.~J."}

MACRO {ieeese} {"IEEE Trans. Software Eng."}

MACRO {ieeetc} {"IEEE Trans. Comput."}

MACRO {ieeetcad}
 {"IEEE Trans. Comput. Aid. Des."}

MACRO {ipl} {"Inf. Process. Lett."}

MACRO {jacm} {"J.~ACM"}

MACRO {jcss} {"J.~Comput. Syst. Sci."}

MACRO {scp} {"Sci. Comput. Program."}

MACRO {sicomp} {"SIAM J. Comput."}

MACRO {tocs} {"ACM Trans. Comput. Syst."}

MACRO {tods} {"ACM Trans. Database Syst."}

MACRO {tog} {"ACM Trans. Graphic."}

MACRO {toms} {"ACM Trans. Math. Software"}

MACRO {toois} {"ACM Trans. Office Inf. Syst."}

MACRO {toplas} {"ACM Trans. Progr. Lang. Syst."}

MACRO {tcs} {"Theor. Comput. Sci."}

FUNCTION {bibinfo.check}
{ swap$
  duplicate$ missing$
    {
      pop$ pop$
      ""
    }
    { duplicate$ empty$
        {
          swap$ pop$
        }
        { swap$
          pop$
        }
      if$
    }
  if$
}
FUNCTION {bibinfo.warn}
{ swap$
  duplicate$ missing$
    {
      swap$ "missing " swap$ * " in " * cite$ * warning$ pop$
      ""
    }
    { duplicate$ empty$
        {
          swap$ "empty " swap$ * " in " * cite$ * warning$
        }
        { swap$
          pop$
        }
      if$
    }
  if$
}

% FUNCTION {format.eprint}
% { eprint duplicate$ empty$
%     'skip$
%     { "\eprint"
%       archive empty$
%         'skip$
%         { "[" * archive * "]" * }
%       if$
%       "{" * swap$ * "}" *
%     }
%   if$  
% }
FUNCTION {format.eprint}
{ eprint empty$
    { "" }
    {
      "\arxivv{" eprint * "}" *
    }
  if$
}

FUNCTION {format.url}
{ url empty$
    { "" }
    {
      "\urll{" url * "}" *
    }
  if$
}

INTEGERS { nameptr namesleft numnames }


STRINGS  { bibinfo}

FUNCTION {format.names}
{ 'bibinfo :=
  duplicate$ empty$ 'skip$ {
  's :=
  "" 't :=
  #1 'nameptr :=
  s num.names$ 'numnames :=
  numnames 'namesleft :=
    { namesleft #0 > }
    { s nameptr
      "{f.~}{vv~}{ll}{, jj}"
      format.name$
      bibinfo bibinfo.check
      't :=
      nameptr #1 >
        {
          nameptr #1
          #1 + =
          numnames #1
          > and
            { "others" 't :=
              #1 'namesleft := }
            'skip$
          if$
          namesleft #1 >
            { "; " * t * }
            {
              s nameptr "{ll}" format.name$ duplicate$ "others" =
                { 't := }
                { pop$ }
              if$
              numnames #2 >
                { ";" * }
                'skip$
              if$
              t "others" =
                {
                  " " * bbl.etal emphasize *
                }
                {
                  bbl.and
                  space.word * t *
                }
              if$
            }
          if$
        }
        't
      if$
      nameptr #1 + 'nameptr :=
      namesleft #1 - 'namesleft :=
    }
  while$
  } if$
}
FUNCTION {format.names.ed}
{
  format.names
}
FUNCTION {format.authors}
{ author "author" format.names
}
FUNCTION {get.bbl.editor}
{ editor num.names$ #1 > 'bbl.editors 'bbl.editor if$ }

FUNCTION {format.editors}
{ editor "editor" format.names duplicate$ empty$ 'skip$
    {
      " " *
      get.bbl.editor
      capitalize
   "(" swap$ * ")" *
      *
    }
  if$
}
FUNCTION {format.isbn}
{ isbn "isbn" bibinfo.check
  duplicate$ empty$ 'skip$
    {
      "ISBN " swap$ *
    }
  if$
}

FUNCTION {format.issn}
{ issn "issn" bibinfo.check
  duplicate$ empty$ 'skip$
    {
      "ISSN " swap$ *
    }
  if$
}

FUNCTION {format.collaboration}
{ collaboration "collaboration" bibinfo.check
  duplicate$ empty$ 'skip$
    {
      "" swap$ *
    }
  if$
}

FUNCTION {format.doi}
{ doi empty$
    { "" }
    {
      "\doi{" doi * "}" *
    }
  if$
}
FUNCTION {format.note}
{
 note empty$
    { "" }
    { note #1 #1 substring$
      duplicate$ "{" =
        'skip$
        { output.state mid.sentence =
          { "l" }
          { "u" }
        if$
        change.case$
        }
      if$
      note #2 global.max$ substring$ * "note" bibinfo.check
    }
  if$
}

FUNCTION {format.title}
{ title
  "title" bibinfo.check
  duplicate$ empty$ 'skip$
    {
      "\enquote{" swap$ *
      "}, " *
    }
  if$
}
FUNCTION {end.quote.title}
{ title empty$
    'skip$
    { before.all 'output.state := }
  if$
}
FUNCTION {output.bibitem}
{ newline$
  "\bibitem{" write$
  cite$ write$
  "}" write$
  newline$
  ""
  before.all 'output.state :=
}

FUNCTION {n.dashify}
{
  't :=
  ""
    { t empty$ not }
    { t #1 #1 substring$ "-" =
        { t #1 #2 substring$ "--" = not
            { "--" *
              t #2 global.max$ substring$ 't :=
            }
            {   { t #1 #1 substring$ "-" = }
                { "-" *
                  t #2 global.max$ substring$ 't :=
                }
              while$
            }
          if$
        }
        { t #1 #1 substring$ *
          t #2 global.max$ substring$ 't :=
        }
      if$
    }
  while$
}

FUNCTION {word.in}
{ "" }

FUNCTION {format.date}
{
  ""
  duplicate$ empty$
  year  "year"  bibinfo.check duplicate$ empty$
    { swap$ 'skip$
        { "there's a month but no year in " cite$ * warning$ }
      if$
      *
    }
    { swap$ 'skip$
        {
          swap$
          " " * swap$
        }
      if$
      *
      remove.dots
    }
  if$
  duplicate$ empty$
    'skip$
    {
      before.all 'output.state :=
    " (" swap$ * ")" *
    }
  if$
}
FUNCTION {format.year}
{ year "year" bibinfo.check duplicate$ empty$
    {  "empty year in " cite$ *
       warning$
    }
    {
  "(" swap$ * ")" *
    }
  if$
}
FUNCTION {format.btitle}
{ title "title" bibinfo.check
  duplicate$ empty$ 'skip$
    {
      emphasize
    }
  if$
}
FUNCTION {either.or.check}
{ empty$
    'pop$
    { "can't use both " swap$ * " fields in " * cite$ * warning$ }
  if$
}
FUNCTION {format.bvolume}
{ volume empty$
    { "" }
    { bbl.volume volume tie.or.space.prefix
      "volume" bibinfo.check * *
      series "series" bibinfo.check
      duplicate$ empty$ 'pop$
        { swap$ bbl.of space.word * swap$
          emphasize * }
      if$
      "volume and number" number either.or.check
    }
  if$
}
FUNCTION {format.number.series}
{ volume empty$
    { number empty$
        { series field.or.null }
        { series empty$
            { number "number" bibinfo.check }
            { output.state mid.sentence =
                { bbl.number }
                { bbl.number capitalize }
              if$
              number tie.or.space.prefix "number" bibinfo.check * *
              bbl.in space.word *
              series "series" bibinfo.check *
            }
          if$
        }
      if$
    }
    { "" }
  if$
}
FUNCTION {is.num}
{ chr.to.int$
  duplicate$ "0" chr.to.int$ < not
  swap$ "9" chr.to.int$ > not and
}

FUNCTION {extract.num}
{ duplicate$ 't :=
  "" 's :=
  { t empty$ not }
  { t #1 #1 substring$
    t #2 global.max$ substring$ 't :=
    duplicate$ is.num
      { s swap$ * 's := }
      { pop$ "" 't := }
    if$
  }
  while$
  s empty$
    'skip$
    { pop$ s }
  if$
}

FUNCTION {convert.edition}
{ extract.num "l" change.case$ 's :=
  s "first" = s "1" = or
    { bbl.first 't := }
    { s "second" = s "2" = or
        { bbl.second 't := }
        { s "third" = s "3" = or
            { bbl.third 't := }
            { s "fourth" = s "4" = or
                { bbl.fourth 't := }
                { s "fifth" = s "5" = or
                    { bbl.fifth 't := }
                    { s #1 #1 substring$ is.num
                        { s eng.ord 't := }
                        { edition 't := }
                      if$
                    }
                  if$
                }
              if$
            }
          if$
        }
      if$
    }
  if$
  t
}

FUNCTION {format.edition}
{ edition duplicate$ empty$ 'skip$
    {
      convert.edition
      output.state mid.sentence =
        { "l" }
        { "t" }
      if$ change.case$
      "edition" bibinfo.check
      " " * bbl.edition *
    }
  if$
}
INTEGERS { multiresult }
FUNCTION {multi.page.check}
{ 't :=
  #0 'multiresult :=
    { multiresult not
      t empty$ not
      and
    }
    { t #1 #1 substring$
      duplicate$ "-" =
      swap$ duplicate$ "," =
      swap$ "+" =
      or or
        { #1 'multiresult := }
        { t #2 global.max$ substring$ 't := }
      if$
    }
  while$
  multiresult
}
FUNCTION {format.pages}
{ pages duplicate$ empty$ 'skip$
    { duplicate$ multi.page.check
        {
          bbl.pages swap$
          n.dashify
        }
        {
          bbl.page swap$
        }
      if$
      tie.or.space.prefix
      "pages" bibinfo.check
      * *
    }
  if$
}
FUNCTION {first.page}
{ 't :=
  ""
    {  t empty$ not t #1 #1 substring$ "-" = not and }
    { t #1 #1 substring$ *
      t #2 global.max$ substring$ 't :=
    }
  while$
}

FUNCTION {format.journal.pages}
{ pages duplicate$ empty$ 'pop$
    { swap$ duplicate$ empty$
        { pop$ pop$ format.pages }
        {
          ": " *
          format.year * " " *
          swap$
          first.page
          pages multi.page.check
            'bbl.pages
            'bbl.page
          if$
          swap$ tie.or.space.prefix
          "pages" bibinfo.check
          * *
          *
        }
      if$
    }
  if$
}
FUNCTION {format.journal.eid}
{ eid "eid" bibinfo.check
  duplicate$ empty$ 'pop$
    { swap$ duplicate$ empty$ 'skip$
      {
          ": " *
      }
      if$
      swap$ *
    }
  if$
}
FUNCTION {format.vol.num.pages}
{ volume field.or.null
  duplicate$ empty$ 'skip$
    {
      bbl.volume swap$ tie.or.space.prefix
      "volume" bibinfo.check
      * *
    }
  if$
  number "number" bibinfo.check duplicate$ empty$ 'skip$
    {
      swap$ duplicate$ empty$
        { "there's a number but no volume in " cite$ * warning$ }
        'skip$
      if$
      swap$
      "~(" swap$ * ")" *
    }
  if$ *
  eid empty$
    { format.journal.pages }
    { format.journal.eid }
  if$
}

FUNCTION {format.chapter.pages}
{ chapter empty$
    'format.pages
    { type empty$
        { bbl.chapter }
        { type "l" change.case$
          "type" bibinfo.check
        }
      if$
      chapter tie.or.space.prefix
      "chapter" bibinfo.check
      * *
      pages empty$
        'skip$
        { ", " * format.pages * }
      if$
    }
  if$
}

FUNCTION {format.booktitle}
{
  booktitle "booktitle" bibinfo.check
  emphasize
}
FUNCTION {format.in.ed.booktitle}
{ format.booktitle duplicate$ empty$ 'skip$
    {
      format.bvolume duplicate$ empty$ 'pop$
        { ", " swap$ * * }
      if$
      editor "editor" format.names.ed duplicate$ empty$ 'pop$
        {
          get.bbl.editor
          swap$ "," *
          " " * swap$ *
          "(" swap$ * ")" *
          swap$
          " " * swap$
          * }
      if$
      word.in swap$ *
    }
  if$
}
FUNCTION {empty.misc.check}
{ author empty$ title empty$ howpublished empty$
  month empty$ year empty$ note empty$
  and and and and and
    { "all relevant fields are empty in " cite$ * warning$ }
    'skip$
  if$
}
FUNCTION {format.thesis.type}
{ type duplicate$ empty$
    'pop$
    { swap$ pop$
      "t" change.case$ "type" bibinfo.check
    }
  if$
}
FUNCTION {format.tr.number}
{ number "number" bibinfo.check
  type duplicate$ empty$
    { pop$ bbl.techrep }
    'skip$
  if$
  "type" bibinfo.check
  swap$ duplicate$ empty$
    { pop$ "t" change.case$ }
    { tie.or.space.prefix * * }
  if$
}
FUNCTION {format.article.crossref}
{
  key duplicate$ empty$
    { pop$
      journal duplicate$ empty$
        { "need key or journal for " cite$ * " to crossref " * crossref * warning$ }
        { "journal" bibinfo.check emphasize word.in swap$ * }
      if$
    }
    { word.in swap$ * " " *}
  if$
  " \cite{" * crossref * "}" *
}
FUNCTION {format.crossref.editor}
{ editor #1 "{vv~}{ll}" format.name$
  "editor" bibinfo.check
  editor num.names$ duplicate$
  #2 >
    { pop$
      "editor" bibinfo.check
      " " * bbl.etal
      emphasize
      *
    }
    { #2 <
        'skip$
        { editor #2 "{ff }{vv }{ll}{ jj}" format.name$ "others" =
            {
              "editor" bibinfo.check
              " " * bbl.etal
              emphasize
              *
            }
            {
             bbl.and space.word
              * editor #2 "{vv~}{ll}" format.name$
              "editor" bibinfo.check
              *
            }
          if$
        }
      if$
    }
  if$
}
FUNCTION {format.book.crossref}
{ volume duplicate$ empty$
    { "empty volume in " cite$ * "'s crossref of " * crossref * warning$
      pop$ word.in
    }
    { bbl.volume
      swap$ tie.or.space.prefix "volume" bibinfo.check * * bbl.of space.word *
    }
  if$
  editor empty$
  editor field.or.null author field.or.null =
  or
    { key empty$
        { series empty$
            { "need editor, key, or series for " cite$ * " to crossref " *
              crossref * warning$
              "" *
            }
            { series emphasize * }
          if$
        }
        { key * }
      if$
    }
    { format.crossref.editor * }
  if$
  " \cite{" * crossref * "}" *
}
FUNCTION {format.incoll.inproc.crossref}
{
  editor empty$
  editor field.or.null author field.or.null =
  or
    { key empty$
        { format.booktitle duplicate$ empty$
            { "need editor, key, or booktitle for " cite$ * " to crossref " *
              crossref * warning$
            }
            { word.in swap$ * }
          if$
        }
        { word.in key * " " *}
      if$
    }
    { word.in format.crossref.editor * " " *}
  if$
  " \cite{" * crossref * "}" *
}
FUNCTION {format.org.or.pub}
{ 't :=
  ""
  year empty$
    { "empty year in " cite$ * warning$ }
    'skip$
  if$
  address empty$ t empty$ and
  year empty$ and
    'skip$
    {
      add.blank "(" *
      t empty$
        { address "address" bibinfo.check *
        }
        { t *
          address empty$
            'skip$
            { ", " * address "address" bibinfo.check * }
          if$
        }
      if$
      year empty$
        'skip$
        { t empty$ address empty$ and
            'skip$
            { " " swap$ * * }
          if$
          year "year" bibinfo.check
          *
        }
      if$
      ")" *
    }
  if$
}
FUNCTION {format.publisher.address}
{ publisher "publisher" bibinfo.warn format.org.or.pub
}

FUNCTION {format.organization.address}
{ organization "organization" bibinfo.check format.org.or.pub
}

FUNCTION {article}
{ output.bibitem
  format.collaboration output
  format.authors "author" output.check
  format.title "title" output.check
  end.quote.title
  crossref missing$
    {
      journal
      "journal" bibinfo.check
      emphasize
      "journal" output.check
      format.vol.num.pages output
    }
    { format.article.crossref output.nonnull
      format.pages output
    }
  if$
  format.issn output
  format.doi output
  % format.note output
  format.eprint output
  fin.entry
}
FUNCTION {book}
{ output.bibitem
  author empty$
    { format.editors "author and editor" output.check
    }
    { format.authors output.nonnull
      crossref missing$
        { "author and editor" editor either.or.check }
        'skip$
      if$
    }
  if$
  format.btitle "title" output.check
  crossref missing$
    { format.bvolume output
      format.number.series output
      format.publisher.address output
    }
    {
      format.book.crossref output.nonnull
      format.date "year" output.check
    }
  if$
  format.edition output
  format.isbn output
  format.doi output
  format.note output
  format.eprint output
  fin.entry
}
FUNCTION {booklet}
{ output.bibitem
  format.authors output
  format.title "title" output.check
  end.quote.title
  howpublished "howpublished" bibinfo.check output
  address "address" bibinfo.check output
  format.date output
  format.isbn output
  format.doi output
  format.note output
  format.eprint output
  fin.entry
}

FUNCTION {inbook}
{ output.bibitem
  author empty$
    { format.editors "author and editor" output.check
    }
    { format.authors output.nonnull
      crossref missing$
        { "author and editor" editor either.or.check }
        'skip$
      if$
    }
  if$
  format.btitle "title" output.check
  crossref missing$
    {
      format.bvolume output
      format.chapter.pages "chapter and pages" output.check
      format.number.series output
      format.publisher.address output
    }
    {
      format.chapter.pages "chapter and pages" output.check
      format.book.crossref output.nonnull
    }
  if$
  format.edition output
  crossref missing$
    { format.isbn output }
    'skip$
  if$
  format.doi output
  format.note output
  format.eprint output
  fin.entry
}

FUNCTION {incollection}
{ output.bibitem
  format.authors "author" output.check
  format.title "title" output.check
  end.quote.title
  crossref missing$
    { format.in.ed.booktitle "booktitle" output.check
      format.number.series output
      format.chapter.pages output
      format.publisher.address output
      format.edition output
      format.isbn output
    }
    { format.incoll.inproc.crossref output.nonnull
      format.chapter.pages output
    }
  if$
  format.doi output
  format.note output
  format.eprint output
  fin.entry
}
FUNCTION {inproceedings}
{ output.bibitem
  format.authors "author" output.check
  format.title "title" output.check
  end.quote.title
  crossref missing$
    { format.in.ed.booktitle "booktitle" output.check
      format.number.series output
      publisher empty$
        { format.organization.address output }
        { organization "organization" bibinfo.check output
          format.publisher.address output
        }
      if$
      format.isbn output
      format.issn output
    }
    { format.incoll.inproc.crossref output.nonnull
    }
  if$
  add.blank
  format.pages "pages" output.check
  format.doi output
  format.note output
  format.eprint output
  fin.entry
}
FUNCTION {conference} { inproceedings }
FUNCTION {manual}
{ output.bibitem
  author empty$
    { organization "organization" bibinfo.check
      duplicate$ empty$ 'pop$
        { output
          address "address" bibinfo.check output
        }
      if$
    }
    { format.authors output.nonnull }
  if$
  format.btitle "title" output.check
  author empty$
    { organization empty$
        {
          address "address" bibinfo.check output
        }
        'skip$
      if$
    }
    {
      organization "organization" bibinfo.check output
      address "address" bibinfo.check output
    }
  if$
  format.edition output
  format.date output
  format.doi output
  format.note output
  format.eprint output
  fin.entry
}

FUNCTION {mastersthesis}
{ output.bibitem
  format.authors "author" output.check
  format.title
  "title" output.check
  end.quote.title
  bbl.mthesis format.thesis.type output.nonnull
  school "school" bibinfo.warn output
  address "address" bibinfo.check output
  format.date "year" output.check
  format.doi output
  format.note output
  format.eprint output
  fin.entry
}

FUNCTION {misc}
{ output.bibitem
  format.collaboration output
  format.authors output
  format.title output
  end.quote.title
  format.date output
  format.doi output
  format.note output
  howpublished "howpublished" bibinfo.check output
  format.eprint output
  fin.entry
  empty.misc.check
}
FUNCTION {phdthesis}
{ output.bibitem
  format.authors "author" output.check
  format.title
  "title" output.check
  end.quote.title
  bbl.phdthesis format.thesis.type output.nonnull
  school "school" bibinfo.warn output
  address "address" bibinfo.check output
  format.date "year" output.check
  format.doi output
  format.note output
  format.eprint output
  fin.entry
}

FUNCTION {proceedings}
{ output.bibitem
  editor empty$
    { organization "organization" bibinfo.check output
    }
    { format.editors output.nonnull }
  if$
  format.btitle "title" output.check
  format.bvolume output
  format.number.series output
  editor empty$
    { publisher empty$
        'skip$
        {
          format.publisher.address output
        }
      if$
    }
    { publisher empty$
        {
          format.organization.address output }
        {
          organization "organization" bibinfo.check output
          format.publisher.address output
        }
      if$
     }
  if$
  format.isbn output
  format.issn output
  format.doi output
  format.note output
  format.eprint output
  fin.entry
}

FUNCTION {techreport}
{ output.bibitem
  format.authors "author" output.check
  format.title
  "title" output.check
  end.quote.title
  format.tr.number emphasize output.nonnull
  institution "institution" bibinfo.warn output
  address "address" bibinfo.check output
  format.date "year" output.check
  format.doi output
  format.note output
  format.url output
  format.eprint output
  fin.entry
}

FUNCTION {unpublished}
{ output.bibitem
  format.authors "author" output.check
  format.title "title" output.check
  end.quote.title
  format.date output
  format.doi output
  format.note "note" output.check
  format.eprint output
  fin.entry
}

FUNCTION {default.type} { misc }
READ
STRINGS { longest.label }
INTEGERS { number.label longest.label.width }
FUNCTION {initialize.longest.label}
{ "" 'longest.label :=
  #1 'number.label :=
  #0 'longest.label.width :=
}
FUNCTION {longest.label.pass}
{ number.label int.to.str$ 'label :=
  number.label #1 + 'number.label :=
  label width$ longest.label.width >
    { label 'longest.label :=
      label width$ 'longest.label.width :=
    }
    'skip$
  if$
}
EXECUTE {initialize.longest.label}
ITERATE {longest.label.pass}
FUNCTION {begin.bib}
{ preamble$ empty$
    'skip$
    { preamble$ write$ newline$ }
  if$
  "\begin{thebibliography}{"  longest.label  * "}" *
  write$ newline$
  "\newcommand{\enquote}[1]{``#1''}"
  write$ newline$
  "\providecommand{\url}[1]{\texttt{#1}}"
  write$ newline$
  "\providecommand{\urlprefix}{URL }"
  write$ newline$
  "\expandafter\ifx\csname urlstyle\endcsname\relax"
  write$ newline$
  "  \providecommand{\doi}[1]{doi:\discretionary{}{}{}#1}\else"
  write$ newline$
  "  \providecommand{\doi}{doi:\discretionary{}{}{}\begingroup \urlstyle{rm}\Url}\fi"
  write$ newline$
  "\providecommand{\bibAnnoteFile}[1]{%"
  write$ newline$
  "  \IfFileExists{#1}{\begin{quotation}\noindent\textsc{Key:} #1\\"
  write$ newline$
  "  \textsc{Annotation:}\ \input{#1}\end{quotation}}{}}"
  write$ newline$
  "\providecommand{\bibAnnote}[2]{%"
  write$ newline$
  "  \begin{quotation}\noindent\textsc{Key:} #1\\"
  write$ newline$
  "  \textsc{Annotation:}\ #2\end{quotation}}"
  write$ newline$
  "\providecommand{\eprint}[2][]{\url{#2}}"
  write$ newline$
}
EXECUTE {begin.bib}
EXECUTE {init.state.consts}
ITERATE {call.type$}
FUNCTION {end.bib}
{ newline$
  "\end{thebibliography}" write$ newline$
}
EXECUTE {end.bib}
%% End of customized bst file
%%
%% End of file `thisthesis.bst'.
