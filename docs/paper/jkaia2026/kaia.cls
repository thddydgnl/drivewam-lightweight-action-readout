\NeedsTeXFormat{LaTeX2e}
\def\kaia@version{1.0}
\def\kaia@filedate{2020/07/02}
\ProvidesClass{kaia}[\kaia@filedate\space v\kaia@version\space LaTeX2e document class]

\RequirePackage[utf8]{inputenc}

\AtBeginDocument{\@ifpackageloaded{amsmath}{\@mathmargin\z@}{}}%
\newif\ifkaia@abstract \kaia@abstractfalse
\newif\ifkaia@uppercase \kaia@uppercasetrue

\DeclareOption{abstract}{\kaia@abstracttrue}%
\DeclareOption{nouppercase}{\kaia@uppercasefalse} %
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}%

\ProcessOptions\relax

\LoadClass[10pt,oneside,a4paper,onecolumn]{article}

\RequirePackage[a4paper, left=20mm, right=20mm, top=30mm, bottom=30mm]{geometry} %
\RequirePackage{times,mathptmx} %
\RequirePackage{fix2col,indentfirst} %
\RequirePackage{ifthen}
\usepackage{setspace}
\usepackage{array,multirow}
\usepackage{hyperref}
\pagenumbering{arabic}

\RequirePackage[nooneline]{caption}
\usepackage{ifpdf}
\ifpdf
    \pdfinfo {
      /Title (KIAI 2020)
      /Subject ()
      /Author (KAIA)
      /Keywords ()
   }
\fi

\usepackage{graphicx} 
  
\columnsep6mm %
\parindent4mm %
\parsep\z@ %

\renewcommand\footnotesize{%
   \@setfontsize\small{9pt}\@ixpt
   \abovedisplayskip 8.5\p@ \@plus3\p@ \@minus4\p@
   \abovedisplayshortskip \z@ \@plus2\p@
   \belowdisplayshortskip 4\p@ \@plus2\p@ \@minus2\p@
   \def\@listi{\leftmargin\leftmargini
               \parsep 0\p@ \@plus1\p@ \@minus\p@
               \topsep 4\p@ \@plus2\p@ \@minus4\p@
               \itemsep0\p@}%
   \belowdisplayskip \abovedisplayskip
}
\def\@openbib@code{\footnotesize}

\frenchspacing %\loppy

\newlength{\@ctmp}
\newlength{\@figindent}
\setlength{\@figindent}{1pc}

\long\def\@makecaption#1#2{
   \setbox\@tempboxa\hbox{\small \noindent #1.~#2}
   \setlength{\@ctmp}{\hsize}
   \addtolength{\@ctmp}{-\@figindent}\addtolength{\@ctmp}{-\@figindent}
   % IF longer than one indented paragraph line
   \ifdim \wd\@tempboxa >\@ctmp
      % THEN DON'T set as an indented paragraph
      {\small #1.~#2\par}
   \else
      % ELSE center
      \hbox to\hsize{\hfil\box\@tempboxa\hfil}
\fi}


\pagestyle{plain}

% maketitle
\renewcommand{\maketitle}{
      \twocolumn[
            \begin{center}
                  \Large\bf
                  \ifkaia@uppercase\MakeUppercase{\kaia@title}\else\kaia@title\fi\\ 
                  \normalfont
                  \ifkaia@abstract
                        \vspace{45pt}
                  \else
                        \kaia@processauthors
                        \vspace{8pt}
                        \vspace{1.5\baselineskip}
                        \large\kaia@author\\
                        \vspace{10pt} 
                        \small\kaia@processaffiliations
                        \kaia@affiliation\\ 
                        \kaia@email 
                        \vspace{2\baselineskip} 
                  \fi
            \end{center}
      ]
}



\newcounter{kaia@asterisks}
\newcommand{\asterisks}[1]{\setcounter{kaia@asterisks}{#1}%
\whiledo{\value{kaia@asterisks}>0}{$^\ast$\addtocounter{kaia@asterisks}{-1}}}

\newcommand{\kaia@title}{}
\renewcommand{\title}[1]{\renewcommand{\kaia@title}{#1}}

\newcounter{kaia@temp}

\newcounter{kaia@affiliations}
\setcounter{kaia@affiliations}{0}

\newcommand{\affiliation}[2]{
      \stepcounter{kaia@affiliations}
      \expandafter\newcount\csname kaia@affiliationcounter#2\endcsname
      \csname kaia@affiliationcounter#2\endcsname=\value{kaia@affiliations}
      \expandafter\def\csname kaia@affiliation\alph{kaia@affiliations}%
      \endcsname{$^{\expandafter\the\csname kaia@affiliationcounter#2\endcsname}$ #1}
}

\newcommand{\kaia@processaffiliations}
{
      \newcommand{\kaia@affiliation}
      {
            \ifthenelse{\value{kaia@authors}>0}
            {
                  \setcounter{kaia@temp}{0}
                  \whiledo{\value{kaia@temp}<\value{kaia@affiliations}}
                  {
                        \stepcounter{kaia@temp}
                        \csname kaia@affiliation\alph{kaia@temp}\endcsname\linebreak
                  }
            }
            {
                  \ClassWarning{kaia}{No affiliations defined}
            }
      }
}

\newcounter{kaia@authors}
\setcounter{kaia@authors}{0}

\renewcommand{\author}[2]{
      \stepcounter{kaia@authors}
      \expandafter\def\csname kaia@author\alph{kaia@authors}\endcsname
      {#1$^{\expandafter\the\csname kaia@affiliationcounter#2\endcsname}$}
}

\newcommand{\kaia@processauthors}
{
\newcommand{\kaia@author}{
      \ifthenelse{\value{kaia@authors}>0}
            {\kaia@authora\setcounter{kaia@temp}{1}%
            \whiledo{\value{kaia@temp}<\value{kaia@authors}}{\stepcounter{kaia@temp}%
            \ifthenelse{\equal{\value{kaia@temp}}{\value{kaia@authors}}}{ and }{, }%
            \csname kaia@author\alph{kaia@temp}\endcsname}}
            {\ClassWarning{kaia}{No author names defined}}}
}

\newcommand{\kaia@email}{\ClassWarning{kaia}{No e-mail address defined}}
\newcommand{\email}[1]{\renewcommand{\kaia@email}{#1}}


\renewenvironment{abstract}{  \center\textbf{\large{Abstract}} \par\vspace{0.5\baselineskip}\begin{quote}}{\end{quote}}

\newenvironment{keywords}{\par\small \begin{quote}\textbf{\textit{Keywords}}---\itshape}{\end{quote}}

% Define Section style
\renewcommand\section{\@startsection {section}{1}{\z@}
                                     {-3.5ex \@plus -1ex \@minus -.2ex}
                                     {2.3ex \@plus .2ex}
                                     {\center \large \textsc}}
                                
\renewcommand\thesection{\roman{section}.}

% Define Subsection style
\renewcommand\subsection{\@startsection {subsection}{2}{\z@}
                                        {-3.25ex \@plus -1 ex \@minus -.2ex}
                                        {1.5ex \@plus .2ex}
                                        {\normalfont\normalsize\itshape}}
\renewcommand\thesubsection{\Alph{subsection}.}

% Define Subsubsection style
\renewcommand\subsubsection{\@startsection {subsubsection}{3}{\parindent}
                                           {1\baselineskip \@plus 0.25\baselineskip \@minus 0.25\baselineskip}
                                           {0ex}
                                           {\itshape}}

\renewcommand\thesubsubsection{\arabic{subsubsection})}


\def\tablename{\normalsize{T}\footnotesize{able}}
\def\figurename{\normalsize{F}\footnotesize{ig.}}

\bibliographystyle{plain}

\endinput
