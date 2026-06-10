**Semantische Segmentierung von Dreiecksnetzen durch Geometrisches Deep Learning auf der Grundlage Nicht-Euklidischer Geometrie** 

## **Masterarbeit im Studiengang Wirtschaftsingenieurwesen** 

## **Friedrich-Alexander-Universität Erlangen-Nürnberg Lehrstuhl für Fertigungsautomatisierung und Produktionssystematik Prof. Dr.-Ing. J. Franke** 

Bearbeiter: Benedikt Scheffler, 22395134 Betreuer: Prof. Dr.-Ing. J. Franke Patrick Bründl, M.Sc., B.Sc., B.A. 

Abgabetermin: 31.12.2022 Bearbeitungszeit: 6 Monate 

## **Erklärung** 

Ich versichere, dass ich die Arbeit ohne fremde Hilfe und ohne Benutzung anderer als der angegebenen Quellen angefertigt habe und dass die Arbeit in gleicher oder ähnlicher Form noch keiner anderen Prüfungsbehörde vorgelegen hat und von dieser als Teil einer Prüfungsleistung angenommen wurde. Alle Ausführungen, die wörtlich oder sinngemäß übernommen wurden, sind als solche gekennzeichnet. 

Nürnberg, den 22.12.2022 

Benedikt Scheffler 

I 

Inhaltsverzeichnis 

## **Inhaltsverzeichnis** 

|**1**|**EINLEITUNG UNDZIELSTELLUNG ............................................................................ 1**|**EINLEITUNG UNDZIELSTELLUNG ............................................................................ 1**|
|---|---|---|
|**2**|**EINFÜHRUNG IN DASMASCHINELLELERNEN ........................................................... 4**||
||**2.1**|**Maschinelles Lernen und Deep Learning ........................................................ 4**|
|||**2.1.1** **Einordnung neuronaler Netze ............................................................................... 4**|
|||**2.1.2** **Künstliches Neuron ................................................................................................ 7**|
|||**2.1.3** **Multilayer Perceptron ............................................................................................. 7**|
||**2.2**|**Bestandteile neuronaler Netze ......................................................................... 9**|
|||**2.2.1** **Optimierungsalgorithmen .................................................................................... 10**|
|||**2.2.2** **Aktivierungsfunktionen........................................................................................ 12**|
|||**2.2.3** **Verlustfunktionen ................................................................................................. 16**|
|||**2.2.4** **Hyperparameter .................................................................................................... 18**|
||**2.3**|**Anwendungsspezifische Netzwerkarchitekturen ...........................................21**|
|||**2.3.1** **Faltende neuronale Netze .................................................................................... 22**|
|||**2.3.2** **Transfer Learning ................................................................................................. 25**|
||**2.4**|**Bewertung und Analyse von ML-Modellen .....................................................26**|
|||**2.4.1** **Bias und Varianz ................................................................................................... 26**|
|||**2.4.2** **Underfitting und Overfitting ................................................................................ 27**|
|||**2.4.3** **Trainings-, Validierungs- und Testfehler ............................................................ 27**|
||**2.5**|**Mathematische und Informationstechnische Grundlagen des Deep**|
|||**Learnings ..........................................................................................................29**|
|||**2.5.1** **Tensoren und Tensorbibliotheken ...................................................................... 30**|
|||**2.5.2** **Automatische Differenzierung und Berechnungsgraphen .............................. 31**|
||**2.6**|**Grundlagen des Geometrischen Deep Learnings ..........................................32**|
|||**2.6.1** **Topologie ............................................................................................................... 32**|
|||**2.6.2** **Einordnung Geometrischen Deep Learnings .................................................... 34**|
|||**2.6.3** **Intrinsische Datenoperationen ............................................................................ 38**|
|**3**|**BESTEHENDEANSÄTZE UNDABLEITUNG DESFORSCHUNGSBEDARFS .................... 40**||
||**3.1**|**Bestehende Ansätze zur Automatisierung in der Verdrahtung von**|
|||**Schaltschränken ..............................................................................................40**|
||**3.2**|**Bestehende Ansätze zur Analyse von Dreiecksnetzen mittels Maschineller**|
|||**Lernverfahren ...................................................................................................41**|
||**3.3**|**Ableitung des Forschungsbedarfs .................................................................41**|
|**4**|**ANGEWANDTEMETHODIK ................................................................................... 43**||



II 

Inhaltsverzeichnis 

||**4.1**|**Data Mining Methodology for Engineering Applications ...............................43**|
|---|---|---|
||**4.2**|**Software-Entwicklung ......................................................................................45**|
|||**4.2.1** **Programmierrichtlinien ........................................................................................ 45**|
|||**4.2.2** **Gewährleistung hochperformanter Programme ............................................... 46**|
|||**4.2.3** **Implikationen der Hardware ................................................................................ 47**|
|**5**|**DATENAGGREGATION, MODELLIERUNG UNDTECHNISCHEIMPLEMENTIERUNG ......... 50**||
||**5.1**|**Datenerhebung .................................................................................................50**|
|||**5.1.1** **Datenaggregation ................................................................................................. 50**|
|||**5.1.2** **Datenvorverarbeitung .......................................................................................... 52**|
||**5.2**|**Labeling der Daten ...........................................................................................61**|
|||**5.2.1** **Dreiecksnetzlabels ............................................................................................... 61**|
|||**5.2.2** **Labels der Graustufenbilder ................................................................................ 63**|
||**5.3**|**Modellierung künstlich neuronaler Netze .......................................................64**|
|||**5.3.1** **MeshCNN ............................................................................................................... 65**|
|||**5.3.2** **MedMeshCNN ........................................................................................................ 67**|
|||**5.3.3** **DiffusionNet .......................................................................................................... 68**|
|||**5.3.4** **YOLOv6 .................................................................................................................. 71**|
|||**5.3.5** **Instanziierung der Segmentierungsmasken ...................................................... 73**|
|||**5.3.6** **Berechnung der Metriken .................................................................................... 75**|
|||**5.3.7** **Inferenzpipeline .................................................................................................... 77**|
||**5.4**|**Optimierung der neuronalen Netze .................................................................80**|
|||**5.4.1** **Hyperparameteroptimierung ............................................................................... 80**|
|||**5.4.2** **Ray Tune ................................................................................................................ 81**|
|**6**|**ERGEBNISSE UNDDISKUSSION ............................................................................ 83**||
||**6.1**|**Datenexploration und Datenverständnis ........................................................83**|
|||**6.1.1** **Strukturierte Analyse der Rohdaten ................................................................... 83**|
|||**6.1.2** **Analyse des Datensatzes ..................................................................................... 86**|
||**6.2**|**Auswertung ......................................................................................................90**|
|||**6.2.1** **Analyse der optimierten Hyperparameter .......................................................... 90**|
|||**6.2.2** **Kennwerte der optimierten Netze ....................................................................... 92**|
|||**6.2.3** **Nutzen und Limitationen der neuronalen Netze .............................................. 103**|
|**7**|**ZUSAMMENFASSUNG UNDAUSBLICK.................................................................. 105**||
|**LITERATURVERZEICHNIS ............................................................................................ 107**|||
|**ANHANGA – AUTOPEP8 KONFIGURATIONSDATEI ........................................................ 116**|||



III 

Inhaltsverzeichnis 

**ANHANG B – DATENSATZINFORMATIONEN .................................................................. 117 ANHANG C – VERWENDETE SOFTWAREVERSIONEN ..................................................... 126 ANHANG D – DIGITALER ANHANG .............................................................................. 128** 

IV 

Abbildungsverzeichnis 

## **Abbildungsverzeichnis** 

Abbildung 1: Exemplarische Identifikation der Position, Größe und Normalvektoren der Merkmale der Schaltschrankkomponenten ....................................................................................... 2 Abbildung 2: Einordnung von ML und DL nach [8] und [11] .................................................................... 5 Abbildung 3: Die Grundlagen und vier Säulen des Maschinellen Lernens nach [13] ............................. 6 Abbildung 4: Aufbau eines künstlichen Neurons nach [18] und [19] ....................................................... 7 Abbildung 5: Topologische Illustration des Aufbaus vollvernetzter neuronaler Netze nach [8] .............. 8 Abbildung 6: Gradient Descent nach [15] .............................................................................................. 10 Abbildung 7: SGD mit und ohne Momentum nach [26] ......................................................................... 12 Abbildung 8: Sigmoid und Ableitung der sigmoid-Aktivierungsfunktion nach [19] ................................ 13 Abbildung 9: Tanh und Ableitung der tanh-Aktivierungsfunktion nach [19] .......................................... 14 Abbildung 10: ReLU und Ableitung der ReLU-Aktivierungsfunktion [19] .............................................. 15 Abbildung 11: Leaky ReLU und Ableitung der Leaky ReLU-Funktion nach [19] .................................. 16 Abbildung 12: Abgrenzung von Klassifikation, Lokalisierung, Segmentierung und Instanzensegmentierung nach [20] .......................................................................... 17 Abbildung 13: Vergleich zwischen der Raster- und Zufallssuche bei der Minimierung einer Funktion mit einem relevanten und einem irrelevanten Parameter nach [36] und [38] ........... 19 Abbildung 14: Qualitative Illustration der Bayes'schen Optimierung am Beispiel einer eindimensionalen Funktion nach [36] .................................................................................................... 21 Abbildung 15: Aufbau eines CNNs nach [41] ........................................................................................ 22 Abbildung 16: Faltungsoperation eines CNN-Blocks nach [41] ............................................................ 23 Abbildung 17: Beispielhafte Faltungsoperation mit Null-Padding (1,1), Ausdehnung (1,1), Kernelgröße (3,3) und Schrittweite (2,2) nach [20] und [33] ......................................................... 24 Abbildung 18: Max- und Average-Pooling nach [8] ............................................................................... 25 Abbildung 19: Bias und Varianz nach [19] ............................................................................................ 27 Abbildung 20: Qualitative Darstellung der Annäherung an Bayes Error über die Zeit nach [19] .......... 29 Abbildung 21: Beispielhafter Berechnungsgraph nach [33] .................................................................. 31 Abbildung 22: Topologische Verformungen verschiedener Mannigfaltigkeiten nach [53] und [54] ...... 34 Abbildung 23: Die 5G des GDL: "Grids, Groups, Graphs, Geodesics & Gauges" nach [52] ................ 35 Abbildung 24: Grundbegriffe der Riemannschen Geometrie am Beispiel der zweidimensionalen Kugel nach [52] ................................................................................................................... 36 Abbildung 25: Paralleltransport des Tangentenvektors A nach C nach [52] ......................................... 37 Abbildung 26: Veranschaulichung des Unterschieds zwischen klassischem und geometrischem CNN nach [51] und [52] ..................................................................................................... 39 Abbildung 27: DMME als holistische Erweiterung des CRISP-DM nach [62] ....................................... 44 Abbildung 28: Graustufenbilder von oben, unten, vorne, hinten, links und rechts der Komponente 1SAE231111M0622 .................................................................................................. 53 Abbildung 29: Einfache Netzunterteilung nach [84] .............................................................................. 56 Abbildung 30: Vergleich zwischen einfacher und glatter Netzunterteilung am Beispiel einer Reihenklemme .......................................................................................................... 57 

V 

Abbildungsverzeichnis 

Abbildung 31: Dichtefunktion der logarithmischen Normalfunktion mit µ = 0 und σ = 0,005 ................ 59 Abbildung 32: Vergleich zwischen normaler Reihenklemme (a), mit Rauschen (b) und mit LaplaceFilter und As-Rigid-As-Possible-Deformationen (c) .................................................. 60 Abbildung 33: Gelabelte Vertices einer Reihenklemme in Blender ...................................................... 62 Abbildung 34: Exemplarische Zuordnung der Scheitelpunkt-Labels zu den Scheitelpunkten .............. 63 Abbildung 35: Vergleich zwischen Knoten-, Kanten-, und Flächenlabels ............................................. 63 Abbildung 36: Exemplarische Labels der Features mittels Bounding-Boxen der Komponte 1SAE231111M0622 .................................................................................................. 64 Abbildung 37: Invariante Faltung auf Dreiecksnetzen nach [61] ........................................................... 66 Abbildung 38: Eingangsmerkmale des MeshCNNs nach [61] .............................................................. 66 Abbildung 39: Mesh Pooling und Mesh Unpooling nach [61] ................................................................ 67 Abbildung 40: DiffusionNet Architektur nach [56] .................................................................................. 70 Abbildung 41: YOLOv6 Architektur nach [98]........................................................................................ 72 Abbildung 42: RepVGG Architektur nach [99] ....................................................................................... 73 Abbildung 43: Zuweisen von Scheitelpunkten an den Grenzen der Merkmale zur Standardklasse .... 74 Abbildung 44: Cluster-, Volumen-, Oberflächenschwerpunkt, sowie Normalvektor einer mittels des DiffusionNet segmentierten Kabeleinführung ........................................................... 76 Abbildung 45: Gesamtinferenz der implementierten Netzwerke ........................................................... 78 Abbildung 46: Prozess zur Abbildung einer 2D-Bounding-Box in den 3D-Raum ................................. 80 Abbildung 47: Häufigkeiten der verschiedenen Dateiformate nach den Konvertierungen je Hersteller 84 Abbildung 48: Histogramm der Dateigrößen aller STEP-Dateien des Datensatzes ............................. 85 Abbildung 49: Skalierung der OBJ-Dateien auf eine fast einheitliche Anzahl an Scheitelpunkten ....... 86 Abbildung 50: Verteilung der Hersteller des gelabelten Datensatzes ................................................... 87 Abbildung 51: Verteilungen der Instanzen der vier Merkmale Kontaktierung, Aufrastpunkt, Kabeleinführung und Beschriftungsfläche für den Trainings- und Testdatensatz .... 88 Abbildung 52: Histogramm der Häufigkeiten der Anzahl an Scheitelpunkten je Merkmal .................... 89 Abbildung 53: Verteilungen der Scheitelpunktanzahl pro Merkmalsinstanz ......................................... 90 Abbildung 54: Trainings- und Validierungs-Loss des DiffusionNets ..................................................... 92 Abbildung 55: Trainings- und Validierungsgenauigkeit des DiffusionNets ............................................ 93 Abbildung 56: Validierungs-Dice-Koeffizienten des DiffusionNets ........................................................ 94 Abbildung 57: Beispielhafte Segmentierung durch das trainierte DiffusionNet der Komponente 0446017 .................................................................................................................... 95 Abbildung 58: Trainings- und Validierungs-Loss des MeshCNN .......................................................... 96 Abbildung 59: Validierungs-Dice-Koeffizienten des MeshCNN ............................................................. 97 Abbildung 60: Validierungs-Loss des YOLOv6-CNN untergliedert in alle Bestandteile der Verlustfunktion .......................................................................................................... 98 Abbildung 61: Validierungs-mAP und mAP über IoU-Schwellwerte zwischen 0,5 und 0,95 ................ 99 Abbildung 62: Exemplarische Darstellung eines problematischen Bauteils für die YOLO-Inferenz am Beispiel der Komponente 1005331 ........................................................................ 100 Abbildung 63: Detektion der Aufrastpunkte durch das YOLO-CNN .................................................... 101 Abbildung 64: Vergleich der Vorhersagen auf von FreeCAD generierten Bildern (links) sowie auf von OCCT generierten Bildern (rechts) ......................................................................... 102 

VI 

Tabellenverzeichnis 

## **Tabellenverzeichnis** 

Tabelle 1: Hardware- und Softwarespezifikationen der NVIDIA GeForce RTX 2080 Ti ....................... 48 Tabelle 2: Hardware- und Softwarespezifikationen des verwendeten PCs .......................................... 48 Tabelle 3: HPO des DiffusionNet .......................................................................................................... 80 Tabelle 4: Gewichte für die verwendeten Verlustfunktionen ................................................................. 91 Tabelle 5: Hyperparameter des besten DiffusionNets .......................................................................... 91 

VII 

Codeverzeichnis 

## **Codeverzeichnis** 

Code 1: Aufbau einer STL-Datei im ASCII-Format nach [83] ............................................................... 54 Code 2: Aufbau einer OBJ-Datei ........................................................................................................... 54 Code 3: Aufbau einer OFF-Datei ........................................................................................................... 54 Code 4: Pseudolabels eines Bauteils mit einer Kontaktierung (1), einem Aufrastpunkt (3), einer Kabeleinführung (0), sowie einer Beschriftungsfläche (2) ........................................ 64 Code 5: Berechnung des aggregierten CUDA-Speichers in Bytes in Python mittels PyTorch nach [33] .................................................................................................................................. 68 

VIII 

Abkürzungsverzeichnis 

## **Abkürzungsverzeichnis** 

|Adam|Adaptive Moment Estimation|
|---|---|
|ANN|Artificial Neural Network|
|API|Application Programming Interface|
|ASCII|American Standard Code for Information Interchange|
|ASHA|Asynchronous Successive Halving Algorithm|
|BCE|Binary-Cross-Entropy|
|BLAS|Basic Linear Algebra Subprograms|
|CAD|Computer Aided Design|
|CE|Cross-Entropy|
|CNN|Convolutional Neural Network|
|COCO|Common Objects in Context|
|CPU|Central Processing Unit|
|CRISP-DM|Cross-Industry Standard Process for Data Mining|
|CUDA|Compute Unified Device Architecture|
|DL|Deep Learning|
|DM|Data Mining|
|DMME|Data Mining Methodology for Engineering Applications|
|DNN|Deep Forward Neural Network|
|e2e|End-to-End|
|ELU|Exponential Linear Unit|
|GB|Gigabyte|
|GDL|Geometrisches Deep Learning|
|GPGPU|General Purpose Computation on Graphics Processing Unit|
|GPU|Graphics Processing Unit|
|HDD|Hard Disk Drive|
|HKS|Heat Kernel Signatur|
|HLP|Human-Level Performance|
|HTTP|Hypertext Transfer Protocol|
|HPO|Hyperparameteroptimierung|
|IoU|Intersection over Union|
|JSON|JavaScript Object Notation|
|KB|Kilobyte|



IX 

Abkürzungsverzeichnis 

|KDD|Knowledge Discovery in Databases|
|---|---|
|KI|Künstliche Intelligenz|
|KMU|kleine und mittlere Unternehmen|
|mAP|mean Average Precision|
|MB|Megabyte|
|ML|Maschinelles Lernen|
|MLP|Multilayer Perceptron|
|NLL|Negative-Logarithmic Likelihood|
|NVCC|Nvidia CUDA Compiler|
|OBJ|Wavefront OBJ|
|OCCT|Open Cascade Technology|
|OFF|Open File Format|
|OS|Operating System|
|PAN|Path Aggregation Network|
|PC|Personal Computer|
|PEP|Python Enhancement Proposal|
|pip|Pip Installs Packages|
|RAM|Random Access Memory|
|ReLU|Rectified Linear Unit|
|Rep|Reparametrization|
|RGB|Rot, Grün, Blau|
|RL|Repräsentatives Lernen|
|RNN|Recurrent Neural Network|
|SELU|Scaled Exponential Linear Unit|
|SEMMA|Sample, Explore, Modify, Model and Assess|
|SGD|Stochastic Gradient Descent|
|STEP|Standard for the Exchange of Product Model Data|
|STL|Standard Triangle Language bzw. Standard Tessellation|
||Language|
|tanh|Tangens hyperbolicus|
|TB|Terabyte|
|TOPS|Tensoroperationen pro Sekunde|
|VGG|Visual Geometry Group|
|YOLO|You Only Look Once|



1 

1 Einleitung und Zielstellung 

## **1 Einleitung und Zielstellung** 

Der globale Wettbewerb führt Unternehmen in der fertigenden Industrie zu immer kürzeren Lieferzeiten und einer hohen Produktvarianz. Über Elektronik und Software soll zusätzlich der Funktionsumfang erhöht werden. [1] Aufgrund steigender Bedarfe im Bereich der Steuerungs- und Energietechnik steigt sowohl die Relevanz als auch der quantitative Bedarf von Schaltanlagen. [2] Entgegen diesem Trend ist die Fertigung von Schaltschränken jedoch durch fehlende Datendurchgängigkeit geprägt. Zusätzlich werden Automatisierungslösungen zwar schrittweise in der Fertigung eingesetzt, diese konnten bisher jedoch nicht die weitgehend händische Arbeit in der Montage ausreichend unterstützen. Die Herstellung eines Schaltschranks wird üblicherweise durch die Erstellung eines Schaltplans durch einen Anlagenhersteller initiiert. Der Hersteller des Schaltschranks montiert nach der Annahme des Auftrags und Bestellung aller benötigten Komponenten dann den Schaltschrank. [3] Die Montage untergliedert sich unter anderem in das Ablängen, Abisolieren, sowie das Anbringen und Crimpen der Aderendhülsen. Während diese vorbereitenden Schritte überwiegend durch Teil- oder Vollautomaten erfolgen, läuft die letztliche Verdrahtung fast ausschließlich manuell ab. Die Verdrahtung der einzelnen Komponenten ist deshalb der zeitintensivste Fertigungsschritt und beansprucht bis zu 49 % der gesamten Montagezeit. [1] Die hohe Komplexität lässt sich in separate Schritte wie das Auslesen und Deuten von Anleitungen, dem Auffinden von Quelle und Ziel sowie dem Anschließen der Drähte an die Klemmstellen untergliedern. [3] Diese für Menschen sehr einfach auszuführenden, jedoch mittels herkömmlicher Robotik schwer zu automatisierenden Schritte, sind ein klassisches Beispiel für das Moravecsche Paradox. [4] Aufgrund der hohen Produktvarianz und damit einhergehenden kleinen Losgröße haben bisherige Ansätze zur Automatisierung dieses Prozesses wenige Erfolge vorzuweisen und beschränken sich häufig auf die rein konzeptionelle Verbesserung der Verdrahtung. [2] Während es bereits wenige technische Lösungen zur Teilautomatisierung gibt, sind diese bisher weder ökonomisch sinnvoll noch für kleine und mittlere Unternehmen (KMU) geeignet. Hinzu kommt, dass bisherige Ansätze überwiegend visuell arbeiten und daher lediglich auf die reale Fertigung ausgelegt sind und die Datendurchgängigkeit des Prozesses deshalb nicht berücksichtigen. [5][6][7] Da der Digitale Zwilling immer wichtiger wird, empfiehlt sich daher ein Verfahren, dessen Resultat sowohl für die Fertigung selbst als auch das digitale Abbild gewinnbringend ist. Diese Arbeit untersucht deshalb Verfahren des Maschinellen Lernens (ML) basierend auf der Geometrie der Schaltschrankkomponenten zur Bestimmung geometrischer Features. Das Forschungsanliegen untergliedert sich daher in die Identifikation der geometrischen Merkmale und Bestimmung der Größe sowie der Ortsvektoren zu diesen. Die geometrischen Features werden als relevant angesehen, sobald Kenntnisse über diese für die Montage nötig sind. Die 

2 

1 Einleitung und Zielstellung 

Betrachtung derjenigen Punkte, welche zur Befestigung der Komponente auf die Tragschiene, der Bestimmung der Beschriftungsflächen, der Kontaktierpositionen der Klemmkammern sowie für in der Montage eingesetzte Werkzeuge wichtig sind, stehen deshalb besonders im Fokus. Ausgehend hiervon ist das Ziel dieser Arbeit die Implementierung und anschließende Evaluation eines Verfahrens zur automatischen Bestimmung der für die Montage relevanten geometrischen Features, basierend auf Deep Learning (DL) ausgehend von Computer Aided Design (CAD) Modellen. Abbildung 1 stellt das finale Ziel der Arbeit dar, indem die Position, Größe und Normalvektoren automatisch identifiziert werden. 

**==> picture [325 x 88] intentionally omitted <==**

**----- Start of picture text -----**<br>
. vas<br>Größe<br>oS \<br>Position (x, y, z)<br>**----- End of picture text -----**<br>


_Abbildung 1: Exemplarische Identifikation der Position, Größe und Normalvektoren der Merkmale der Schaltschrankkomponenten_ 

Zu Beginn der Arbeit werden zunächst die Grundlagen der Arbeit erläutert. Diese unterteilen sich in sechs Bereiche. Das erste Kapitel liefert die nötigen Grundlagen für ML und DL, indem neuronale Netze zuerst eingeordnet werden und dann das grundlegende Funktionsprinzip zum einen mittels eines künstlichen Neurons sowie dem Aufbau tiefer neuronaler Netze erklärt wird. Danach folgt die Erläuterung aller wesentlichen Bestandteile neuronaler Netze. Diese umfassen verschiedene Optimierungsalgorithmen, Aktivierungsfunktionen, Verlustfunktionen, Schichten und sonstige Hyperparameter. Darauf aufbauend folgt eine Erklärung anwendungsspezifischer Aspekte neuronaler Netze. Diese umfassen Merkmale faltender neuronaler Netze, Transfer Learning und einer abschließenden Erklärung des Zusammenhangs zwischen der Netzwerkarchitektur und dem tatsächlichen Inferenzziel. In Kapitel 2.4 folgt die Bewertung und Analyse von ML-Modellen, welche Bias und Varianz, Underfitting und Overfitting, sowie Trainings-, Validierungs- und Testfehler betrachtet. Das nächste Kapitel liefert die nötigen mathematischen und informationstechnischen Grundlagen des Deep Learnings. Hierfür wird auf Matrizen 

3 

1 Einleitung und Zielstellung 

und Tensoren, Computation Graphs und Automatische Differenzierung eingegangen. Die Grundlagen werden durch einen Einblick in das Geometrische Deep Learning (GDL) abgeschlossen. Hierfür wird zuerst auf die Topologielehre eingegangen, um Aspekte Nicht-Euklidischer Geometrie und von Dreiecksoberflächennetzen zu beleuchten. Für das Verständnis der Symmetrieeigenschaften selbst und dem Grundgedanken des GDL wird dieses zudem in vier Gruppen klassifiziert und abschließend die Motivation intrinsischer Datenoperationen erläutert. 

Ausgehend von bestehenden Ansätzen zur automatisierten Analyse von Schaltschrankkomponenten und Ansätzen zur Analyse von Dreiecksnetzen mittels ML, folgt die Ableitung des Forschungsbedarfs. Zur Gewährleistung der Replizierbarkeit und guter wissenschaftlicher Praxis baut diese Arbeit auf der Data Mining Methodology for Engineering Applications (DMME) auf und beleuchtet darüber hinaus den Software-Entwicklungszyklus der Arbeit, bei welchem insbesondere auf die Gewährleistung hochperformanter Software als auch deren Implikationen für die Hardware eingegangen wird. Danach wird die Datenvorverarbeitung, Modellierung und Technische Implementierung betrachtet. Hierfür wird auf die Datenaggregation mittels Web-Scraping und die Datenvorverarbeitung eingegangen, bevor dann das Labeln der zwei- und dreidimensionalen Daten erläutert wird. Hiernach werden verschiedene Implementierungen neuronaler Netze erklärt und abschließend durch die angewandten Verfahren der Hyperparameteroptimierung abgerundet. Die Diskussion beschäftigt sich anschließend mit der Exploration und dem Verständnis der Daten, bestehend aus einer strukturierten Analyse der Rohdaten sowie der Auswahl der Daten. Die Auswertung analysiert zuerst die optimierten Hyperparameter und Kennwerte der optimierten Netze und evaluiert dann verschiedene ClusteringAlgorithmen und die gesamte End-to-End (e2e) Inferenz. Nach der gesamtheitlichen Betrachtung der Ergebnisse dieser Arbeit werden diese mit bereits bestehenden Benchmarks und anderen Anwendungsfällen der verwendeten Netzwerkarchitekturen verglichen. Die Arbeit schließt mit einem Fazit und Ausblick auf weitere Forschungsarbeiten ab. 

4 

2 Einführung in das Maschinelle Lernen 

## **2 Einführung in das Maschinelle Lernen** 

Entgegen der Erwartung fallen nach dem Moravecschen Paradox für die Ausführung sensormotorischer und wahrnehmungsfähiger Aufgaben wesentlich mehr Rechenressourcen an als für logisches Denken. [4] Vor diesem Hintergrund lässt sich erklären, dass abstrakte und formale Aufgaben, welche für Menschen zu den schwierigsten geistigen Leistungen zählen, für einen Computer sehr einfach umzusetzen sind. Da ein Großteil aller Informationen subjektiv und intuitiv vorliegt, besteht eine wesentliche Herausforderung darin, Verfahren zu erarbeiten, welche dem Computer ermöglichen dieses Wissen zu erfassen, um sich intelligent verhalten zu können. Der aktuelle Stand der Technik ermöglicht mittlerweile auch das Erlernen einer, dem Menschen ähnlichen, Intuition, wie beispielsweise das Erkennen von Gegenständen, durch Methoden des ML, auf welche nun nachfolgend eingegangen wird. [8] 

## **2.1 Maschinelles Lernen und Deep Learning** 

Aufgrund der Neuheit des Begriffs ML ist eine einheitliche Definition für diesen in der Literatur nicht vorhanden. Trotz des aktuellen Trends wird an maschinellen Lernverfahren schon seit den 1950er Jahren geforscht. Der Begriff ML ist erstmals durch A. L. Samuel geprägt worden, welcher diesen als das Aufdecken von Zusammenhängen durch Maschinen bezeichnet, ohne diese vorher explizit zu kennen. [9] 

## **2.1.1 Einordnung neuronaler Netze** 

Im Gegensatz zur Definition nach [9] gibt es mittlerweile treffendere Definitionen, welche den aktuellen Stand der Technik besser beschreiben. Demnach bezeichnet ML den Entwurf und die Analyse von Algorithmen, die es Computern ermöglichen automatisch zu lernen und die es Maschinen ermöglichen aus der automatischen Analyse von Daten Regeln aufzustellen sowie diese zur Vorhersage unbekannter Daten zu nutzen. [10] Abbildung 2 ordnet DL, Repräsentatives Lernen (RL), ML und Künstliche Intelligenz (KI) ein. Es wird deutlich, dass die Begrifflichkeiten hierarchisch miteinander in Verbindung stehen und jeweils spezifischere Teilbereiche repräsentieren. 

5 

2 Einführung in das Maschinelle Lernen 

**==> picture [454 x 285] intentionally omitted <==**

**----- Start of picture text -----**<br>
Künstliche<br>Intelligenz<br>Maschinelles<br>Lernen<br>Repräsentatives<br>Lernen<br>Deep Learning<br>**----- End of picture text -----**<br>


_Abbildung 2: Einordnung von ML und DL nach [8] und [11]_ 

Zusammenfassend gilt es festzuhalten, dass sich dieses Forschungsgebiet der Informatik mit der Entwicklung und Bewertung von Algorithmen befasst, die es Computern ermöglichen, aus Erfahrungen zu lernen. Das Konzept der Erfahrung als Datensatz historischer Ereignisse bietet sich an, um nützliche Informationen in diesem zu identifizieren. Jeder Algorithmus des ML erhält einen Datensatz als Eingabe und liefert ein Modell zur Kodierung der Informationen als Ausgabe. [12] 

ML lässt sich bezüglich der angewandten mathematischen Verfahren unterteilen. Die vier Säulen des ML aus Abbildung 3 sind demnach Regression, Dimensionalitätsreduktion, Dichteschätzung und Klassifikation. Alle diese Verfahren bauen auf Vektoranalysis, analytischer Geometrie, Optimierung, linearer Algebra, Wahrscheinlichkeiten und Verteilungen und Matrixdekompositionen auf. In der linearen Regression besteht das Ziel darin, Funktionen zu finden, die die Eingangswerte auf die entsprechenden beobachteten Funktionswerte abbilden. Das Hauptziel der Dimensionalitätsreduktion besteht darin, eine kompakte Darstellung geringer Dimensionalität von hochdimensionalen Daten zu finden. Bei der Dichteschätzung wird versucht eine Wahrscheinlichkeitsverteilung zur Beschreibung eines Datensatzes ausfindig zu machen. Die Klassifikation als letzte Säule des ML versucht ähnlich wie die Regression Eingaben auf vorgegebene Ausgaben abzubilden. Anders als die Regression arbeitet die Klassifikation jedoch mit ganzzahligen statt reellwertigen Zahlen. [13] 

6 

2 Einführung in das Maschinelle Lernen 

**==> picture [451 x 247] intentionally omitted <==**

**----- Start of picture text -----**<br>
Maschinelles Lernen<br>Dimensionalitäts-<br>Regression Dichteschätzung Klassifizierung<br>reduktion<br>Vektoranalysis Analytische Geometrie Optimierung<br>Lineare Algebra Wahrscheinlichkeiten & Verteilungen Matrixdekomposition<br>**----- End of picture text -----**<br>


## _Abbildung 3: Die Grundlagen und vier Säulen des Maschinellen Lernens nach [13]_ 

Das Forschungsgebiet des DL unterteilt sich wiederum in die Subkategorien unüberwachtes, überwachtes und bestärkendes Lernen. Bestärkendes Lernen basiert auf der Interaktion eines Agenten mit der Umwelt. Die Handlungen des Agenten werden auf Basis des Belohnungsprinzip gemessen und wird im Rahmen des Trainingsprozess maximiert. [14] Das grundlegende Prinzip aus Erfahrung zu lernen ist dabei allgegenwärtig in allen Kategorien des DL. Gegensätzlich zum bestärkenden Lernen werden beim überwachten Lernen dagegen signifikante Informationen durch beschriftete Trainingsbeispiele erlangt und auf einem separaten Datensatz getestet. Das gewonnene Wissen kann dann zur Vorhersage neuer Daten genutzt werden. Beim unüberwachten Lernen wird nicht zwischen Trainings- und Testdaten unterschieden. Stattdessen läuft die Verarbeitung der Eingabedaten unter dem Ziel der Zusammenfassung dieser Daten ab. Die komprimierte Darstellung enthält optimalerweise lediglich Wissen über relevante Merkmale, um anschließend generalisierte Vorhersagen treffen zu können. [15] Der aktuelle Stand der Technik nimmt zudem zunehmend hybride Ansätze wie selbstüberwachtes Lernen in den Fokus, bei welchem entweder globale invariante oder lokale Merkmale erzeugt werden. Der Trainingsprozess findet dann auf den erzeugten Merkmalen statt. [16][17] Unabhängig von den Methoden dieser Domänen kann DL auch als Datenkompression interpretiert werden. Das Ziel besteht immer in der Reduktion der Dimensionalität der Daten. [15] Diese Arbeit beschäftigt sich insbesondere mit Methoden des überwachten Lernens. 

7 

2 Einführung in das Maschinelle Lernen 

## **2.1.2 Künstliches Neuron** 

Ein künstliches Neuron, oftmals in der Literatur auch als Perceptron bezeichnet, ist von der biologischen Nervenzelle inspiriert. Abbildung 4 zeigt den Aufbau eines einzelnen künstlichen Neurons. Es akzeptiert mehrere Inputvariablen und generiert einen Outputwert. Die Inputvariablen sind dabei über die Gewichte mit dem eigentlichen Neuron verbunden. Höhere Gewichte bilden hierbei die Analogie zu stärker ausgeprägten Dendriten in der Biologie. Durch die Multiplikation der Inputvariablen mit den Gewichten und der anschließenden Summation dieser Ergebnisse wird ein Zwischenoutput generiert. Dieser bildet nach dem Addieren eines Bias den Inputwert für eine Aktivierungsfunktion. Diese nichtlineare Funktion ist deshalb relevant, um auch nicht-lineare Funktionen abbilden zu können und spielt insbesondere bei komplexeren Netzwerken bestehend aus mehreren künstlichen Neuronen eine wesentliche Rolle. [18] 

**==> picture [452 x 198] intentionally omitted <==**

**----- Start of picture text -----**<br>
x0 w0<br>σ<br>x1 w1<br>h<br>y<br>Aktivierungsfunktion<br>xn wn<br>Input Gewichte<br>**----- End of picture text -----**<br>


_Abbildung 4: Aufbau eines künstlichen Neurons nach [18] und [19]_ 

## **2.1.3 Multilayer Perceptron** 

Deep Forward Neural Networks (DNN) oder auch als Multilayer Perceptrons (MLP) bezeichnet sind die grundlegenden Modelle des DL. Das Ziel eines DNNs liegt in der Approximation einer unbekannten Funktion durch das Erlernen der Parameter der Funktion. Modelle dieser Art werden als DNN oder MLP bezeichnet, da die Informationen durch die ausgewertete Funktion während den Zwischenberechnungen mehrere Schichten durchlaufen. [20] Im einfachsten Fall gibt es keine Rückkopplungsverbindungen, bei denen Ausgänge späterer mit vorherigen Schichten rückgekoppelt werden. Rückgekoppelte Netzwerkarchitekturen werden als Recurrent Neural Network (RNN) bezeichnet und sind nicht Bestandteil dieser Arbeit. [8] 

8 

2 Einführung in das Maschinelle Lernen 

Aufbauend auf dem vorherigen Kapitel liegt der Unterschied zu einem einzelnen Neuron bei DNNs zum einen in der Verwendung vieler Neuronen in einer Schicht, als auch dem Weiterleiten der Informationen durch mehrere Schichten im DNN. [20] Abbildung 5 zeigt den beispielhaften Aufbau eines DNNs. Hierbei wird deutlich, dass ein MLP theoretisch aus beliebig vielen Inputwerten 𝑥 , verdeckten Schichten sowie Outputwerten 𝑦 bestehen kann. [8] 

**==> picture [452 x 244] intentionally omitted <==**

**----- Start of picture text -----**<br>
Inputschicht Verdeckte Schichte(n) Ausgabeschicht<br>x0<br>y0<br>x1<br>y1<br>x2<br>**----- End of picture text -----**<br>


_Abbildung 5: Topologische Illustration des Aufbaus vollvernetzter neuronaler Netze nach [8]_ 

Ein DNN unterscheidet sich von einem Artificial Neural Network (ANN) ausschließlich in der Anzahl der verdeckten Schichten. ANN ist hierbei ein übergeordneter Begriff, welcher alle Arten von neuronalen Netzen umfasst. DNNs umfassen dagegen ausschließlich neuronale Netze mit mindestens zwei verdeckten Schichten. Jedes in Abbildung 5 dargestellte Neuron ist hierbei identisch aufgebaut zum künstlichen Neuron aus Abbildung 4. Der einzige Unterschied liegt darin, dass der Ouput eines Neurons wieder als Input für die nächste Schicht verwendet wird. [19] Abhängig von der Anzahl der Neuronen in der aktuellen Schicht und der vorherigen Schicht −1 ergeben sich ⋅( −1) Verbindungen zwischen den einzelnen Neuronen, welche in einer Gewichtsmatrix dargestellt werden können. [20] 

Wie Abbildung 5 veranschaulicht, fließen die Inputwerte von links nach rechts durch das Netzwerk. Dieses Zusammensetzen einzelner Funktionen durch einen gerichteten azyklischen Graphen ermöglicht das Erstellen einer komplexeren Funktion wie in Formel 2.1 veranschaulicht: [19] 

**==> picture [320 x 15] intentionally omitted <==**

9 

2 Einführung in das Maschinelle Lernen 

𝑓[(1)] , 𝑓[(2)] und 𝑓[(3)] stellen hier exemplarisch Funktionen aus den aus Abbildung 5 zugehörigen Schichten dar. Jede Funktion 𝑓[(𝑖)] setzt sich dabei entsprechend Formel 2.2 zusammen 

**==> picture [301 x 14] intentionally omitted <==**

Hierbei repräsentiert 𝜑(⋅) eine beliebige differenzierbare Nichtlinearität. [19] Der Nichtlinearität kommt im MLP eine besondere Bedeutung zu, da ohne diese die gesamte Funktion des neuronalen Netzes zu einer einzigen linearen Funktion kollabieren würde. Als Konsequenz daraus könnte der Output mittels einer einzigen linearen Transformation des Inputs generiert werden. Folglich bildet ein neuronales Netz ohne Nichtlinearitäten keine Möglichkeit zur universalen Funktionsapproximation. [8] 

Zwischen der Input- und Outputschicht können sich beliebig viele verdeckte Schichten befinden. Da die Trainingsdaten im Rahmen des überwachten Lernens mit einem Label versehen sind, besteht das Ziel darin 𝑓(𝑥) so anzupassen, dass es 𝑓[∗] (𝑥) entspricht. Die Trainingsdaten bestehen daher aus verrauschten, approximativen Beispielen für 𝑓[∗] (𝑥) , welche an verschiedenen Trainingspunkten ausgewertet werden. Jedes Beispiel 𝑥 wird von einem Label 𝑦≈𝑓[∗] (𝑥) annotiert. [8] Diese Trainingsbeispiele in Kombination mit den zugehörigen Labels steuern das Verhalten der Ausgabeschicht, indem ein Wert erzeugt werden muss, welcher möglichst nah an 𝑦 liegt. Abgesehen von der Ausgabeschicht, wird das Verhalten der verdeckten Schichten jedoch nicht direkt durch die Trainingsdaten spezifiziert. Der Lernalgorithmus entscheidet lediglich, wie die jeweiligen Schichten genutzt werden sollen. Die Trainingsdaten sehen jedoch nicht spezifische Funktionsweisen für die Schichten vor. Aufgrund der Tatsache, dass die Trainingsdaten nicht die gewünschte Ausgabe für jede dieser Schichten zeigen, werden sie als versteckte Schichten bezeichnet. [19] 

Nach einem vollständigen Durchlaufen aller Berechnungen im neuronalen Netz liegen ein oder mehrere Outputs 𝑦 vor. Als Maß zur Bestimmung wie gut die Vorhersage des Vorwärtsdurchlaufs war, wird nun der Verlust berechnet, welcher auch als Loss bezeichnet wird. Hierfür können verschiedenste Verlustfunktionen verwendet werden, welche abhängig von der Problemstellung Vor- und Nachteile mit sich bringen. Der Output der Verlustfunktion gibt dabei ausschließlich eine Auskunft über den Grad der Optimierung mit Bezug auf die Modellparameter, ist jedoch nicht als allgemeingültige Metrik zur Beurteilung des finalen Ergebnisses zu verstehen. [8] 

## **2.2 Bestandteile neuronaler Netze** 

Nachdem alle grundlegenden Bestandteile eines neuronalen Netzes beleuchtet sind, bedarf es einen Überblick über Parameteroptionen für verschiedene 

10 

2 Einführung in das Maschinelle Lernen 

Anwendungsfälle und Situationen. Bei der Konzeptionierung jedes neuronalen Netzes muss daher ein geeigneter Optimierungsalgorithmus, Aktivierungsfunktion, Schichtarten und weitere Hyperparameter gewählt werden. Nachfolgend werden etablierte Alternativen dieser Konzepte vorgestellt. 

## **2.2.1 Optimierungsalgorithmen** 

Die Approximation einer Funktion mittels DL erfolgt in der Regel über einen langen und schrittweisen Trainingsprozess [19]. Eine Iteration über den gesamten Datensatz wird als Epoche bezeichnet und es bedarf vieler Epochen zur Erzielung zufriedenstellender Ergebnisse. Während einer Epoche werden die Parameter des neuronalen Netzes ständig verändert mit dem Ziel der Übereinstimmung der Vorhersagen mit den tatsächlichen Labels. [20] Die Art und Weise zur Veränderung der Parameter in die korrekte Richtung fällt in den Bereich statistischer Optimierungsverfahren. Die Grundlage der Optimierungsverfahren bildet dabei Gradient-Descent, welcher in unterschiedlichen Ausführungen in nahezu jedem neuronalen Netz zur Optimierung verwendet wird. [21] 

## **Gradient Descent** 

Gradient Descent ist ein Verfahren zur Minimierung einer Zielfunktion 𝐽(𝜃) , die durch die Parameter 𝜃∈ ℝ[𝑑] eines Modells bestimmt ist. Die Optimierung erfolgt durch die Aktualisierung der Parameter in der gegensätzlichen Richtung des Gradienten der Zielfunktion 𝛻𝜃𝐽(𝜃) . Die Lernrate 𝜂 dient zur Bestimmung der Größe der Schritte zur Erreichung eines lokalen Minimums. Abbildung 6 veranschaulicht dieses triviale Verfahren. [21] 

**==> picture [454 x 158] intentionally omitted <==**

**----- Start of picture text -----**<br>
𝐽(𝜃)<br>Initiales Gewicht Gradient<br>Inkrementelle<br>Schritte<br>𝜃<br>**----- End of picture text -----**<br>


_Abbildung 6: Gradient Descent nach [15]_ 

Es gibt drei verschiedene Varianten von Gradient Descent, die sich in der verwendeten Datenmenge zur Berechnung des Gradienten der Zielfunktion unterscheiden. Abhängig von der Datenmenge muss deshalb ein Kompromiss 

11 

2 Einführung in das Maschinelle Lernen 

zwischen der Genauigkeit der Parameteraktualisierung und der für diese benötigte Zeit gefunden werden. Das herkömmliche Batch Gradient-Descent Verfahren berechnet den Gradienten mit Bezug auf alle Parameter 𝜃 über den gesamten Trainingsdatensatz nach Formel 2.3: [21] 

**==> picture [299 x 12] intentionally omitted <==**

Batch Gradient-Descent kann deshalb sehr langsam sein und ist insbesondere für Datensätze, die nicht komplett in den Arbeits- oder Grafikspeicher passen nicht geeignet. Der Vorteil liegt in der garantierten Konvergenz zum globalen Minimum konvexer Funktionen und zu einem lokalen Minimum für nicht-konvexe Funktionen. [22] Eine realistischere Alternative hierzu ist Stochastic Gradient Descent (SGD), welcher die Parameter nach jedem Beispiel im Datensatz entsprechend Formel 2.4 aktualisiert: [21] 

**==> picture [322 x 14] intentionally omitted <==**

SGD vermeidet die redundante Gradientenberechnung ähnlicher Beispiele, indem alle Aktualisierungen im Datensatz nacheinander erfolgen. Aufgrund der hochfrequenten Parameteraktualisierungen schwankt die Annäherung zur Zielfunktion jedoch stärker. [22] Diese teils starken Schwankungen können durch das Mini-Batch Gradient Descent Verfahren aus Formel 2.5 vermieden werden, indem die Aktualisierung des Gradienten weder auf dem gesamten Datensatz noch auf einem einzigen Beispiel basiert. Stattdessen wird der Gradient auf einem Batch an Beispielen berechnet. Die Größe des Batchs entspricht typischerweise einer Zweierpotenz. [21] 

**==> picture [339 x 14] intentionally omitted <==**

Mittels Mini-Batch Gradient Descent kann zum einen die Varianz der Parameteraktualisierungen verringert und zum anderen die Berechnung durch optimierte Tensoroperationen durchgeführt werden. Eine gute Konvergenz mittels Mini-Batch Gradient Descent ist dennoch aufgrund mehrerer Komplikationen nicht garantiert. Die Wahl einer geeigneten Lernrate kann schwierig sein. Es gilt hier zwischen einer kleinen Lernrate mit langsamer Konvergenz und großen, um das Minimum schwankenden, Lernrate abzuwägen. Über Lernratenpläne kann das Anpassen der Lernrate zu vordefinierten Zeitpunkten oder ab dem Erreichen einer Zielsetzung definiert werden. Durch die Definition vorab können jedoch die Eigenschaften des Datensatzes nicht miteinbezogen werden. Zudem gilt für alle Parameteraktualisierungen die gleiche Lernrate. Dies ist insbesondere bei spärlichen und unausgewogenen Daten nicht wünschenswert. [21] Zuletzt besteht eine Herausforderung darin während der Minimierung nicht-konvexer Funktionen nicht in suboptimalen lokalen Minima oder Sattelpunkten festzustecken. Dies führt anderenfalls zu einer drastischen Verlangsamung des Lernprozesses. [23] 

12 

2 Einführung in das Maschinelle Lernen 

## **Gradient Descent Optimierungsalgorithmen** 

Trotz der Vielzahl unterschiedlicher Optimierungsalgorithmen für Gradient Descent empfiehlt der aktuelle Stand der Technik die ausschließliche Verwendung des Adaptive Moment Estimation (Adam) Optimierungsverfahrens. Weitere Algorithmen wie Momentum, Adagrad, Adadelta und RMSprop finden dagegen selten Anwendung, da Adam die Prinzipien dieser vereinigt. [24][25] Adam berechnet zusätzlich zu adaptiven Lernraten für jeden einzelnen Parameter auch den exponentiell abklingenden Durchschnitt der vergangenen quadrierten Gradienten wie bei Adadelta und RMSprop. Darüber hinaus speichert es einen exponentiell abklingenden Durschnitt der vergangenen Gradienten ähnlich wie Momentum. [22] Abbildung 7 stellt das Prinzip eines Momentums dar, welches die Parameteraktualisierungen in die korrekte Richtung beschleunigt und Oszillationen dämpft [26]. SGD ohne Momentum SGD mit Momentum <> C> _Abbildung 7: SGD mit und ohne Momentum nach [26]_ 

Aufgrund der kombinierten Beurteilung der Parameteraktualisierung basierend auf diesen Werten, optimiert Adam in der Praxis besser als andere Optimierungsalgorithmen. [27] Eine detaillierte Erklärung der einzelnen Algorithmen findet sich in [21]. 

## **2.2.2 Aktivierungsfunktionen** 

Aktivierungsfunktionen stellen eine Schlüsselkomponente im Entwurf von ANNs dar. Aufgrund der hohen Anzahl an möglichen Nichtlinearitäten werden nachfolgend die am häufigsten verwendeten Funktionen erläutert und deren Eigenschaften aufgezeigt. [12][20] Ursprünglich wurde fast ausschließlich die Sigmoid-Funktion aus Formel 2.6, sowie deren Ableitung (2.7) verwendet. Abbildung 8 stellt diese dar. [20] 

**==> picture [378 x 69] intentionally omitted <==**

13 

2 Einführung in das Maschinelle Lernen 

**==> picture [454 x 283] intentionally omitted <==**

**----- Start of picture text -----**<br>
1<br>0.9<br>0.8<br>0.7<br>0.6<br>0.5<br>0.4<br>0.3<br>0.2<br>0.1<br>0<br>-10 -8 -6 -4 -2 0 2 4 6 8 10<br>z →<br>sigmoid(z) sigmoid'(z)<br>y →<br>**----- End of picture text -----**<br>


_Abbildung 8: Sigmoid und Ableitung der sigmoid-Aktivierungsfunktion nach [19]_ 

Der Tangens hyperbolicus (tanh) ist ähnlich wie die Sigmoidfunktion eine hilfreiche Alternative. Bei diesem wird deutlich, wie aus Formel 2.8, 2.9 und Abbildung 9 hervorgeht, dass alle Werte im Intervall (-1; 1) liegen. [12] 

**==> picture [302 x 27] intentionally omitted <==**

**==> picture [322 x 27] intentionally omitted <==**

14 

2 Einführung in das Maschinelle Lernen 

**==> picture [427 x 239] intentionally omitted <==**

**----- Start of picture text -----**<br>
1<br>0.8<br>0.6<br>0.4<br>0.2<br>0<br>-10 -8 -6 -4 -2 0 2 4 6 8 10<br>-0.2<br>-0.4<br>-0.6<br>-0.8<br>-1<br>z →<br>y →<br>**----- End of picture text -----**<br>


tanh(z) tanh'(z) 

## _Abbildung 9: Tanh und Ableitung der tanh-Aktivierungsfunktion nach [19]_ 

Auch wenn beide vorherige Nichtlinearitäten funktionieren, bilden diese nicht die eigentliche Funktionsweise eines natürlichen Neurons ab, welches ein Signal entweder weitergibt oder nicht. Als Konsequenz legt der aktuelle Stand der Technik oftmals die Verwendung einer Rectified Linear Unit (ReLU) nahe. Wie Formel 2.10 und 2.11 zeigen, ist diese standardmäßig lediglich über den positiven Teil des Arguments definiert. ReLUs weißen insbesondere für sehr tiefe neuronale Netze bessere Trainingseigenschaften auf, weshalb sie in der aktuellen Literatur fast ausschließlich verwendet werden. Abbildung 10 stellt die normale ReLU-Funktion dar. [28] 

**==> picture [349 x 67] intentionally omitted <==**

15 

2 Einführung in das Maschinelle Lernen 

**==> picture [454 x 283] intentionally omitted <==**

**----- Start of picture text -----**<br>
1.1<br>1<br>0.9<br>0.8<br>0.7<br>0.6<br>0.5<br>0.4<br>0.3<br>0.2<br>0.1<br>0<br>-1 -0.8 -0.6 -0.4 -0.2 0 0.2 0.4 0.6 0.8 1<br>z →<br>relu(z) relu'(z)<br>y →<br>**----- End of picture text -----**<br>


## _Abbildung 10: ReLU und Ableitung der ReLU-Aktivierungsfunktion [19]_ 

ReLU bezeichnet nicht nur eine einzige Aktivierungsfunktion, sondern eine gesamte Gruppe, welche darauf aufbauend erprobt wurden. Die bekannteste Alternative zur herkömmlichen ReLU ist Leaky ReLU aus Formel 2.12 und 2.13, welche eine kleine Steigung für negative Koeffizienten anstelle keiner Steigung aufweist. Der Steigungskoeffizient wird vor dem Training festgelegt und zählt nicht zu den erlernenden Parametern des Netzwerks. Leaky ReLU findet vor allem dann Anwendung, um dem Problem verschwindender Gradienten im Zuge der Fehlerrückführung entgegenzuwirken. Die in Abbildung 11 dargestellte leicht positive Steigung für negative z-Werte, kann beim Problem sehr kleiner Gradienten helfen. [29] 

**==> picture [354 x 67] intentionally omitted <==**

16 

2 Einführung in das Maschinelle Lernen 

**==> picture [454 x 283] intentionally omitted <==**

**----- Start of picture text -----**<br>
1.1<br>1<br>0.9<br>0.8<br>0.7<br>0.6<br>0.5<br>0.4<br>0.3<br>0.2<br>0.1<br>0<br>-1 -0.8 -0.6 -0.4 -0.2 0 0.2 0.4 0.6 0.8 1<br>-0.1<br>-0.2<br>z →<br>lrelu(z) lrelu'(z)<br>y →<br>**----- End of picture text -----**<br>


## _Abbildung 11: Leaky ReLU und Ableitung der Leaky ReLU-Funktion nach [19]_ 

Abgesehen von Leaky ReLU finden sich noch weitere Variationen der ReLU in der Literatur. Hierzu zählen unter anderem die Exponential Linear Unit (ELU) und Scaled Exponential Linear Unit (SELU). Aufgrund der höheren Komplexität von Leaky ReLU, ELU und SELU wird meist die normale ReLU verwendet. Die Berücksichtigung der Komplexität einer Nichtlinearität ist relevant, da in der Fehlerrückführung eben diese differenziert werden muss. [29] 

## **2.2.3 Verlustfunktionen** 

Abhängig vom Inferenzziel eines neuronalen Netzes wird nach dem Forward-Pass der Loss berechnet. Die Berechnung des Verlusts hängt maßgeblich von der Problemstellung ab. Im Bereich visueller DL-Verfahren werden grundlegend vier Problemstellungen unterschieden. Diese unterteilen sich in Klassifizierung, Lokalisierung, Segmentierung und Instanz-Segmentierung. Der Komplexitätsgrad nimmt hierbei zu, da die Konzepte aufeinander aufbauen. Aus Abbildung 12 wird ersichtlich, dass im Rahmen der Klassifizierung der Input auf einen einzigen Wert reduziert wird. Dieser wird als Zahl kodiert und kann nach der Inferenz einer semantischen Bedeutung zugeordnet werden. Bei Lokalisierungsaufgaben wird nicht nur ein Objekt klassifiziert, sondern ebenfalls im Input lokalisiert. Im Zuge dieser Problemstellung wurden ebenfalls Verfahren zur Lokalisierung und Klassifizierung mehrerer Objekte in einem Bild erarbeitet. Zur Detektion mehrerer Objekte in einem 

17 

2 Einführung in das Maschinelle Lernen 

Bild hat sich das Region-Proposal-Verfahren etabliert. [30] Diese neuronalen Netze sind unter dem Namen „You Only Look Once” (YOLO) bekannt. [31] Die Segmentierung kann ebenfalls als Klassifizierung auf Datenpunktebene verstanden werden. Segmentierungsaufgaben sind maßgeblich durch U-Net Architekturen beeinflusst, welche aus einem Encoder und Decoder bestehen. Wie Abbildung 12 zeigt, werden bei der Segmentierung einzelne Segmentierungsmasken je Klasse erzeugt. Die Unterscheidung in mehrere Instanzen innerhalb einer Segmentierungsmaske ist nur über Instanz-Segmentierung möglich. Die Untergliederung in mehrere Instanzen kann sowohl über das Sliding-WindowSystem, Anchor-Boxes oder Clustering-Algorithmen erfolgen. Die Rechenintensität der Verfahren legt eine Verwendung von Anchor-Boxes oder cluster-basierten Methoden nahe. Der erfolgreichste Ansatz wird als panoptische Segmentierung bezeichnet und stellt eine Kombination aus der Segmentierung und InstanzSegmentierung dar, bei welcher alle Datenpunkte sowohl einer Klasse zugeordnet werden als auch die zugehörige Instanz identifiziert wird. Die aufgeführten Verfahren sind auf Rot, Grün, Blau (RGB) Bildern erprobt und setzen sich schrittweise für komplexere Datenstrukturen durch. Für Graphen und Dreiecksnetze ist die InstanzSegmentierung Bestand aktueller Forschung. [32] 

**==> picture [454 x 160] intentionally omitted <==**

**----- Start of picture text -----**<br>
Instanz-<br>Klassifikation Lokalisierung Segmentierung<br>segmentierung<br>Bounding-Box  Segmentierungsmaske<br>Klasse 4 Instanzen<br>Koordinaten auf Pixelbasis<br>**----- End of picture text -----**<br>


_Abbildung 12: Abgrenzung von Klassifikation, Lokalisierung, Segmentierung und Instanzensegmentierung nach [20]_ 

Nach der Identifikation der Problemstellung erfolgt die Wahl und Implementierung einer geeigneten Verlustfunktion. Für Binär-Klassifizierungsaufgaben wird meist der Binary-Cross-Entropy (BCE) Loss verwendet. Problemstellungen, welche eine Unterteilung in mehr als zwei Klassen erfordern, verwenden meist den allgemeinen Cross-Entropy (CE) Loss. Zur Lokalisierung von Objekten eignet sich der YOLOLoss. Zur Segmentierung und Instanz-Segmentierung können der NegativeLogarithmic-Likelihood (NLL) Loss oder ebenfalls CE-Loss verwendet werden. Das Erproben der Eignung unterschiedlicher Verlustfunktionen ist ein wesentlicher Bestandteil der Forschung im DL. Demnach gibt es etliche Verlustfunktionen, die sich 

18 

2 Einführung in das Maschinelle Lernen 

als hilfreich herausgestellt haben. Für einen Überblick geeigneter Verlustfunktionen wird auf [33] und [34] verwiesen. 

## **2.2.4 Hyperparameter** 

Durch die enorm hohe Rechenintensität moderner ML-Algorithmen wird zum einen dedizierte Hardware sowie verteilte Berechnungen zum Erzielen hoher Leistungen in einer angemessenen Zeit benötigt. Das Ziel beim Trainieren neuronaler Netze besteht deshalb nicht nur im Erreichen bestmöglicher Metriken, sondern gleichfalls in einer möglichst effizienten Umsetzung sowie einer beschleunigten Zielerreichung. [35] Aufgrund der Abhängigkeit neuronaler Netze von einer Vielzahl verschiedener Hyperparameter, müssen diese problemorientiert optimiert werden. Dieser Zusammenhang wird als Hyperparameteroptimierung (HPO) oder Hyperparametertuning bezeichnet. Es bestehen bereits viele Algorithmen zur Effizienzsteigerung der Modellauswahl, welche sich in der Rechenintensität und dem abgedeckten Merkmalsraum unterscheiden. Insbesondere sehr einflussreiche Hyperparameter, die einen Einfluss auf die Architektur haben oder für Regularisierung und Optimierungsalgorithmen verantwortlich sind, müssen sorgfältig ermittelt werden. Die Domäne jedes Hyperparameters kann dabei jeweils verschieden sein und reellwertige, binäre oder kategorische Werte annehmen. Bei ganzzahligen und zwangsläufig bei reellwertigen Hyperparametern wird der zu evaluierende Hyperparameterraum dabei eingegrenzt. [36] 

## **Gittersuche** 

Die Gitter- oder Rastersuche ist die einfachste HPO-Methode. Diese auch als vollfaktorieller Versuchsplan bezeichnete Methode basiert lediglich auf der strukturellen Rastersuche des kartesischen Produkts einer zuvor definierten finiten Menge. Der Nachteil dieses Ansatzes besteht in der Anzahl der Dimensionalitäten, da die erforderliche Anzahl von Funktionsauswertungen exponentiell mit der Dimensionalität des Konfigurationsraums wächst. Ein weiteres Problem dieses Ansatzes besteht im starken Anstieg der benötigten Funktionsauswertungen im Zuge einer Erhöhung der Diskretisierungsauflösung. [36][19] 

## **Zufallssuche** 

Eine einfache Alternative zur Gittersuche ist die Zufallssuche, bei welcher die Hyperparameterauswahl rein zufallsbasiert erfolgt. Die Zufallssuche ist insbesondere dann geeigneter als die Rastersuche, wenn einige Parameter wesentlich wichtiger sind als andere. [19][37] Bei einem Budget 𝐵 fester Funktionsauswertungen beträgt die Anzahl der verschiedenen Werte, die die Gittersuche für jeden der 𝑁 Hyperparameter auswählen kann, nur 𝐵[1] ⁄[𝑁] , wohingegen die Zufallssuche 𝐵 

19 

2 Einführung in das Maschinelle Lernen 

verschiedene Werte für jeden Parameter untersucht. Abbildung 13 zeigt ein Beispiel, in welchem die Zufallssuche das geeignetere beider Verfahren ist. [36] 

**==> picture [454 x 199] intentionally omitted <==**

**----- Start of picture text -----**<br>
Rastersuche Zufallssuche<br>Wichtiger Parameter Wichtiger Parameter<br>Unwichtiger Parameter Unwichtiger Parameter<br>**----- End of picture text -----**<br>


_Abbildung 13: Vergleich zwischen der Raster- und Zufallssuche bei der Minimierung einer Funktion mit einem relevanten und einem irrelevanten Parameter nach [36] und [38]_ 

Weitere Vorteile der Zufalls- gegenüber der Gittersuche liegen in der einfachen Parallelisierung und flexiblen Ressourcenzuweisung. Die Zufallssuche ist demnach eine nützliche Grundlage, da sie aufgrund einer fehlenden Annahme über den zu optimierenden Algorithmus bei ausreichenden Funktionsauswertungen eine beliebig nah am Optimum liegende Leistung erzielt. [38] Durch die Verknüpfung der Zufallssuche mit komplexeren Suchstrategien wird zudem eine minimale Konvergenzrate garantiert und fügt eine Exploration zur Verbesserung der modellbasierten Suche hinzu. Darüber hinaus bietet sich die Zufallssuche noch in der Initialisierung des Suchprozesses an, da sie den gesamten Konfigurationsraum abdeckt. Neben diesen Vorteilen stellt die Zufallssuche jedoch keine allgemeingültige Lösung dar. Oftmals sind komplexere Optimierungsverfahren geeigneter. [36] 

## **Bayes’sche Optimierung** 

Die Bayes’sche Optimierung ist ein Verfahren für die globale Optimierung teurer Black-Box-Funktionen. Dieses Optimierungsverfahren aus Formel 2.14 stellt den aktuellen Stand der Technik der HPO dar und erzielt sehr gute Ergebnisse in Problemstellungen verschiedenster Domänen. [36] Der Algorithmus besteht aus einem probabilistischen Ersatzmodell und einer Erfassungsfunktion zur Entscheidung, welcher Punkt als nächstes ausgewertet wird. In jeder Iteration des Algorithmus wird das stellvertretende Modell an die bis dahin aufgezeichneten Beobachtungen angepasst. [19] Die Erfassungsfunktion bestimmt dann unter 

20 

2 Einführung in das Maschinelle Lernen 

Verwendung der Vorhersageverteilung des probabilistischen Modells den Nutzen der verschiedenen Kandidatenpunkte. Im Vergleich zur Auswertung der teuren BlackBox-Funktion ist die Erfassungsfunktion billig zu berechnen und ermöglicht daher eine ausführliche Optimierung. Obwohl viele Erfassungsfunktionen existieren, wird meist die „Erwartete Verbesserung” angewendet, da sie in geschlossener Form berechnet werden kann, wenn die Modellvorhersage 𝑦 bei der Konfiguration 𝜆 einer Normalverteilung wie in Formel 2.15 folgt. [36] 

**==> picture [409 x 55] intentionally omitted <==**

Hierbei stellt 𝜙(⋅) und 𝛷(⋅) die Standard-Dichtefunktion und StandardNormalverteilung dar und 𝑓𝑚𝑖𝑛 den besten bisher beobachteten Wert. Abbildung 14 veranschaulicht die Bayes’sche Optimierung am Beispiel einer eindimensionalen Funktion. Das Ziel liegt in der Minimierung der gestrichelten Zielfunktion unter Verwendung eines Gauß-Prozesses, indem die Erfassungsfunktion maximiert wird. Der Erfassungswert ist nahe der Beobachtungen niedrig. Der höchste Erfassungswert liegt in (a) in einem Punkt, an dem der Wert der vorhergesagten Funktion niedrig und die Vorhersageunsicherheit relativ hoch ist. Während in (b) links der neuen Beobachtung noch eine große Varianz vorhanden ist, ist der vorhergesagte Mittelwert rechts davon sehr niedrig. Obwohl um den Ort des wahren Maximums in (c) fast keine Unsicherheit mehr besteht, wird die nächste Auswertung aufgrund der erwarteten Verbesserung gegenüber dem bisher besten Punkt dort durchgeführt. [39] 

21 

2 Einführung in das Maschinelle Lernen 

**==> picture [454 x 424] intentionally omitted <==**

**----- Start of picture text -----**<br>
Zielfunktion<br>Beobachtung<br>Maximum<br>(a)<br>Erfassungsfunktion<br>(b)<br>Standardabweichung Mittelwert<br>(c)<br>**----- End of picture text -----**<br>


_Abbildung 14: Qualitative Illustration der Bayes'schen Optimierung am Beispiel einer eindimensionalen Funktion nach [36]_ 

## **2.3 Anwendungsspezifische Netzwerkarchitekturen** 

Obwohl normale ANNs aufgrund des universalen Approximationstheorems theoretisch jede Funktion erlernen können, ist dies aus rein praktischen Gründen nicht möglich [39]. Unabhängig von der tatsächlichen Lernkapazität eines ANNs empfiehlt sich die Messung der Leistung anhand der Rechenintensität. Diese wird maßgeblich durch die durchgeführten mathematischen Operationen beeinflusst, weshalb es die Anwendung sehr rechtenintensiver Kalkulationen zu vermeiden gilt. Aufgrund dessen haben sich unterschiedliche Netzwerkarchitekturen etabliert, welche für konkrete Anwendungsfälle geeignet sind. [40] Nachfolgend wird auf 

22 

2 Einführung in das Maschinelle Lernen 

gesonderte Netzwerke eingegangen, sowie Verfahren zur Reduktion der Rechenintensität erläutert. 

## **2.3.1 Faltende neuronale Netze** 

Convolutional Neural Networks (CNN) bezeichnen eine spezielle Art neuronaler Netze, welche sich für die Verarbeitung gitterartiger Topologien eignen. Abbildung 15 stellt den grundlegenden Aufbau eines CNNs dar. Ein CNN-Block setzt sich dabei aus einer oder mehreren Faltungen und einer abschließenden Pooling-Operation zusammen. Am Ende der CNN-Blöcke werden diese in eine flache Schicht ausgerollt und durchlaufen ein vollvernetztes DNN, bevor eine problemorientierte Verlustfunktion berechnet wird. [8] 

**==> picture [454 x 218] intentionally omitted <==**

**----- Start of picture text -----**<br>
Feature Maps<br>Input DNN<br>Convolution Pooling<br>Feature Extraktion Klassifikation bzw.<br>Regression<br>Output<br>**----- End of picture text -----**<br>


## _Abbildung 15: Aufbau eines CNNs nach [41]_ 

LeCun weist darauf hin, dass das Grundprinzip von CNNs darin besteht die Anzahl der freien Modellparameter möglichst weit zu reduzieren, ohne die Rechenleistung des Netzes wesentlich zu beeinträchtigen. [42] Die Anwendung dieses sehr grundlegenden Prinzips erhöht die Wahrscheinlichkeit einer korrekten Verallgemeinerung, da es zu einer spezialisierten Netzarchitektur führt, die eine geringere Entropie aufweist. [43] Der Kernbestandteil eines CNNs ist die Faltungsoperation, deren Funktionsweise in Abbildung 16 dargestellt wird. Der Input wird hierbei mit einem Kernel gefaltet. Während der Faltung iteriert der Kernel schrittweise über den Inputtensor. Zuerst wird der entsprechende Teilbereich des Inputtensors elementweise mit dem Kernel multipliziert und anschließend das Resultat der Multiplikation durch den Plus-Reduktions-Operator auf einen einzelnen Wert reduziert. Diese Schritte werden über alle Bereiche wiederholt, bis der gesamte Inputtensor abgearbeitet wurde. Der Output, welcher nach dem Anwenden der 

23 

2 Einführung in das Maschinelle Lernen 

Faltungsoperation auf den gesamten Inputtensor entsteht, wird als Feature Map bezeichnet. Üblicherweise werden zahlreiche verschiedene Filter verwendet, wobei jeder Filter immer eine Feature Map generiert. Im Falle von zweidimensionalen Inputdaten entstehen demnach dreidimensionale Feature Maps. Die Filteroperation wird jedoch immer auf alle Kanäle angewandt. Aufgrund dieser Eigenschaften entspricht die Dimensionalität der Inputtensoren nicht der der Feature Maps. 

## _Abbildung 16: Faltungsoperation eines CNN-Blocks nach [41]_ 

Da die Faltungsoperation von mehreren Parametern abhängt, müssen alle diese in der Berechnung der Dimensionalität der Feature Maps berücksichtigt werden. Die Dimensionsgröße des Outputs ist dabei durch Formel 2.16 definiert. [33] 

**==> picture [380 x 37] intentionally omitted <==**

mit: 

i Dimension 

D Dimensionsgröße 

p Auffüllung (engl. padding) 

- d Ausdehnung (engl. dilation) 

- k Kernelgröße (engl. kernel size) s Schrittweite (engl. stride) 

24 

2 Einführung in das Maschinelle Lernen 

Abbildung 17 veranschaulicht die Funktionsweise der Parameter Auffüllung, Ausdehnung, Kernelgröße und Schrittweite. Die Parameter können sich in jeder Dimension unterscheiden, weshalb Formel 2.16 auch separat für jede Dimension berechnet werden muss. [33] Padding kann die Verkleinerung der Feature Maps verhindern. Zum einen kann die Größe des Paddings variiert werden und zum anderen die Werte, welche zum Auffüllen genutzt werden. Mögliche Variationen sind beispielsweise Null-Padding, reflektiertes, repliziertes und zirkuläres Padding. Die Ausdehnung der Faltung bestimmt den Abstand zwischen den Kernelelementen. Zusätzlich kann die Größe des Kernels in jeder Dimension geändert werden. Die Schrittweite bestimmt die Rate wie der Kernel über den Inputtensor iteriert. [20] 

**==> picture [454 x 227] intentionally omitted <==**

**----- Start of picture text -----**<br>
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0<br>0 3 0 1 2 2 1 0 0 0 3 0 1 2 2 1 0 0<br>0 2 6 2 4 0 2 1 0 0 2 6 2 4 0 2 1 0<br>0 2 4 1 3 0 1 1 0 0 2 4 1 3 0 1 1 0<br>0 3 0 1 3 7 1 1 0 0 3 0 1 3 7 1 1 0<br>0 3 6 8 4 2 0 7 0 0 3 6 8 4 2 0 7 0<br>0 3 9 1 3 0 4 6 0 0 3 9 1 3 0 4 6 0<br>0 3 8 7 4 0 1 1 0 0 3 8 7 4 0 1 1 0<br>0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0<br>(a) (b)<br>**----- End of picture text -----**<br>


_Abbildung 17: Beispielhafte Faltungsoperation mit Null-Padding (1,1), Ausdehnung (1,1), Kernelgröße (3,3) und Schrittweite (2,2) nach [20] und [33]_ 

Es gilt zu beachten, dass zwar alle Parameter variierbar sind, jedoch im Rahmen eines CNNs vor Beginn des Trainings festgelegt werden. Wie in Abbildung 15 dargestellt, werden üblicherweise eine oder mehrere Faltungen und anschließend eine Pooling-Operation ausgeführt. Durch dieses Downsampling werden die Merkmalskarten in nicht überlappende Rechtecke unterteilt. Am häufigsten wird hierfür Max-Pooling verwendet, bei welchem die jeweiligen Maximalwerte ausgewählt werden. Eine Alternative hierzu ist das Average-Pooling, bei welchem der Durchschnitt aller im Rechteck befindlichen Werte kalkuliert wird. Die Größe der Rechtecke ist ein Hyperparameter, welcher bei der Umsetzung des DNNs festgelegt werden kann. [8] Abbildung 18 veranschaulicht das Funktionsprinzip der PoolingSchichten. Das Ziel einer solchen Pooling-Operation besteht in der Vermeidung der Überanpassung durch das ausschließliche Weitergeben der wichtigsten Daten an die nächste Schicht. Im Kontext visueller Anwendungen hilft die Interpretation des 

25 

2 Einführung in das Maschinelle Lernen 

Lernens sehr einfacher Merkmale in frühen Schichten und zunehmend komplexeren Merkmalen durch das Aggregieren vieler einfacher Merkmale. Die Aggregation wird durch Pooling durchgeführt. [44] 

**==> picture [454 x 94] intentionally omitted <==**

**----- Start of picture text -----**<br>
0 3 1 0<br>4 2 Max Pooling 4 1 2 1 Average Pooling 2 1<br>4 0 3 4 0 0 3 0<br>2 3 0 0<br>**----- End of picture text -----**<br>


## _Abbildung 18: Max- und Average-Pooling nach [8]_ 

Eine Schicht eines neuronalen Netzes, welche die Faltungs- und Pooling-Operation kombiniert wird als CNN-Block bezeichnet, wobei meist mehrere CNN-Blöcke hintereinander durchlaufen werden. [44] 

## **2.3.2 Transfer Learning** 

Transfer Learning ist eine Technik, bei der das bei einer Aufgabe erlernte Wissen zur Lösung einer anderen, ähnlichen Aufgabe genutzt wird. Üblicherweise werden die erfolgreichsten Modelle auf Millionen von Datenpunkten trainiert. Da dies wenig zugänglich ist und state-of-the-art Ergebnisse ohne einen großen Datensatz kaum zu erreichen sind, wird auf die erlernten Parameter vorhandener Modelle zurückgegriffen. [45] Im Kontext von CNNs werden etliche Kernel des Modells beispielsweise auf einem sehr großen Datensatz wie ImageNet trainiert [46]. Da die Filter für eine sehr große Variation von Formen, Farben und Texturen in den Bildern aktiviert werden und gut generalisieren, können diese Filter wiederverwendet werden, um Features für einen neuen Datensatz zu lernen. Hierfür werden die vorderen Schichten eines vorhandenen ML-Modells eingefroren, sodass die Parameter dort nicht mehr während der Fehlerrückführung verändert werden. Im Anschluss daran kann das Netzwerk beliebig erweitert werden. Typischerweise werden am Ende verdeckte Schichten hinzugefügt, um das Anwenden der Features auf den neuen Datensatz zu erlernen. [20] 

Zur erfolgreichen Anwendung von Transfer Learning müssen folgende sechs Schritte durchlaufen werden: [20] 

- Normalisieren des Inputs mit Mittelwert und der Standardabweichung, welche beim Training des vortrainierten Modells verwendet wurde. 

- Laden der Architektur und Gewichte des vortrainierten Modells. 

- Verwerfen der letzten Schichten des vortrainierten Modells. 

26 

2 Einführung in das Maschinelle Lernen 

- Verbinden des gekürzten neuronalen Netzes mit neu initialisierten Schichten. Die Anzahl der Neuronen der letzten Schicht entspricht der Anzahl an Ausgaben des neuronalen Netzes. 

- Sicherstellen, ausschließlich die Parameter der neu initialisierten Schichten trainierbar zu machen. 

- Trainieren des Modells. 

Während die Integration bereits vortrainierter Netze fast ausschließlich mit Vorteilen verbunden ist, besteht die Herausforderung besonders im Finden geeigneter vortrainierter Gewichte. Für herkömmliche Modellarchitekturen stellt dies in der Regel kein Problem dar. Insbesondere in der Erforschung neuer Modellarchitekturen mangelt es jedoch oftmals an vortrainierten Netzen. Ebenfalls hängt die Verbreitung der zur Verfügung stehenden Modelle vom Bedarf, Einsatz und Popularität des Modells ab. Ebenfalls macht das Nutzen eines vortrainierten Netzes keinen Sinn, wenn die Komplexität des eigenen neuartigen Problems ohnehin wesentlich höher ist. [20] Im Falle der in der Arbeit vorgestellten geometrischen Netze liegen zum einen wenige frei verfügbare Netze vor und zum anderen sind diese sowohl in der Komplexität hinsichtlich der Geometrie als auch der Auflösung mit Bezug auf die Anzahl der Datenpunkte nicht hilfreich. Die Verwendung vortrainierter Gewichte macht deshalb keinen Sinn, da die Komplexität insbesondere von den in frühen Netzwerkschichten erkannten Merkmalen abhängt. Ein auf lediglich ein spezifisches Themengebiet trainiertes neuronales Netz ist deshalb meist aufgrund mangelnder Generalisierung nicht ausreichend. [45] 

## **2.4 Bewertung und Analyse von ML-Modellen** 

Vor der Bereitstellung eines ML-Modells wird dieses zuerst evaluiert. Abhängig von der Problemstellung bieten sich unterschiedliche Metriken zur Evaluation an. [19][20] Dieses Kapitel stellt deshalb grundlegende zu beachtende Konzepte in der Evaluierung von ML-Modellen dar. 

## **2.4.1 Bias und Varianz** 

Der Bias ist ein Maß für den Wahrheitsgehalt der Messungen, und die Varianz wie weit diese Messungen um den Durchschnitt abweichen. Aus Abbildung 19 wird deutlich, dass bei sowohl hoher Varianz als auch hohem Bias die Varianz zwar um den Mittelwert gestreut ist, jedoch dieser selbst schon vom tatsächlichen Wert abweicht. [19] 

27 

2 Einführung in das Maschinelle Lernen 

**==> picture [454 x 299] intentionally omitted <==**

**----- Start of picture text -----**<br>
Niedrige Varianz Hohe Varianz<br>Geringe Bias<br>Hohe Bias<br>**----- End of picture text -----**<br>


_Abbildung 19: Bias und Varianz nach [19]_ 

## **2.4.2 Underfitting und Overfitting** 

Die Essenz des Overfittings besteht in der unwissentlichen Konzentration auf denjenigen Teil der Varianz, welcher nicht die zugrunde liegende Datenstruktur darstellt. Durch diese Überanpassung auf den Trainingsdatensatz sinkt deshalb die Vorhersagegenauigkeit auf unbekannten Daten. Wenn das Modell dagegen nicht in der Lage ist die Struktur der Daten zu erfassen, wird von Underfitting gesprochen. Sowohl Underfitting als auch Overfitting ist unerwünscht bei der Beurteilung von Näherungsmodellen für die Datenanalyse. [47][19] 

Das Problem von Underfitting und Overfitting in DNNs liegt darin, dass während des Trainings keine Möglichkeit besteht die Ergebnisse einfach auf Under- und Overfitting zu überprüfen. Für eine objektive Beurteilung bedarf es deshalb einer Unterteilung der Trainingsdaten. [19] 

## **2.4.3 Trainings-, Validierungs- und Testfehler** 

Der gesamte Datensatz wird für eine vollständige und objektive Bewertung in die drei separaten Trainings-, Validierungs- und Testdatensätze aufgeteilt. Während die Aufteilung in diese drei Datensätze eindeutig ist, variiert die Namensgebung in der 

28 

2 Einführung in das Maschinelle Lernen 

Literatur. Neben dem Train/Val/Test-Split wird häufig auch vom Train/Dev/Test-Split oder Train/Dev/Val-Split gesprochen. [8] Im Rahmen dieser Arbeit wird die erste Bezeichnung verwendet. Der Validierungsdatensatz wird entsprechend auch als Development-Datensatz bezeichnet. Das Modell wird auf dem Trainingsdatensatz unter Verwendung der Inputs und deren zugehörigen Labels mittels eines Optimierungsalgorithmus trainiert. Das trainierte Modell wird dann auf dem Validierungsdatensatz angewendet. Durch aussagekräftige Metriken kann Aufschluss über die tatsächliche Performanz des Modells gegeben werden. Der Validierungsdatensatz wird demnach für die Erprobung verschiedener Hyperparameter verwendet. [12] Der Testdatensatz wird zusätzlich für das einmalige Testen des fertigen Modells auf unbekannten Daten und zur Kalkulation der finalen Metriken verwendet. Der Validierungsdatensatz ist zum Testen nicht ausreichend, da durch die Auswahl der optimalen Hyperparameter und Methoden wie beispielsweise Early Stopping eine Voreingenommenheit gegenüber dem Validierungsdatensatz auftritt. [19] Beim Aufsplitten der Daten in die drei Datensätze muss aufgrund der beschriebenen Abhängigkeiten darauf geachtet werden, die Daten nicht zu vermischen. Anderenfalls verlieren sämtliche Metriken ihre Aussagekraft und spiegeln meist ein besseres Ergebnis wider, als tatsächlich vorliegt. [20] 

## **Bayes Error** 

Da Verfahren des überwachten Lernens Labels erfordern, versucht der Optimierungsalgorithmus eine Funktion zu approximieren, deren Output möglichst nah an den Labels liegt. Unter der Voraussetzung, dass die Labels zu 100 % korrekt sind und keine Fehler aufweisen, beträgt die maximal mögliche Genauigkeit ebenfalls 100 %. [8] Komplett fehlerfreie Beschriftungen sind allerdings nahezu ausgeschlossen. Dies kann daran liegen, dass Daten unscharf sind, Flüchtigkeitsfehler unterlaufen, Mehrdeutigkeiten auftreten oder manche Zusammenhänge für den Menschen nicht erkennbar sind. Die maximal erzielbare Genauigkeit korreliert daher mit der Qualität der Labels. Der Beweis, dass der Bayes Error tatsächlich das Minimum ist, basiert auf dem Bayes-Klassifikator, welcher die Wahrscheinlichkeit einer Fehlklassifikation ausgibt. Im Kontext des Bayes Error ist das Heranziehen der Leistung auf dem Niveau des Menschen hilfreich. Dieser als Human-Level Performance (HLP) bezeichnete Wert beschreibt den niedrigsten durch eine Person erreichbaren Wert innerhalb einer Klassifizierungsaufgabe und wird nachfolgend als 𝜖ℎ𝑙𝑝 bezeichnet. 𝜖ℎ𝑙𝑝 repräsentiert immer den niedrigsten durch Menschen erreichbaren Wert. Abbildung 20 verdeutlicht, dass HLP bei geeigneten Problemstellungen schnell erreicht werden kann. Der Bayes Error kann dagegen nur angenähert werden, wobei das Verhältnis der Genauigkeit zur investierten Trainingszeit stark abnimmt. [19] 

29 

2 Einführung in das Maschinelle Lernen 

**==> picture [454 x 169] intentionally omitted <==**

**----- Start of picture text -----**<br>
100<br>100 −<br>𝐵𝑎𝑦𝑒𝑠<br>100 −<br>80 ℎ𝑙𝑝<br>60<br>40<br>20<br>Zeit<br>Genauigkeit<br>**----- End of picture text -----**<br>


_Abbildung 20: Qualitative Darstellung der Annäherung an Bayes Error über die Zeit nach [19]_ 

Da 𝜖ℎ𝑙𝑝 lediglich stichprobenartig geschätzt, nicht jedoch genau bestimmt werden kann, dient der Wert nur als Richtwert. Da Menschen durch gemeinsame Zusammenarbeit an einem Problem teilweise bessere Ergebnisse erzielen können, definiert sich HLP in der Literatur auch als der geringstmögliche Fehler, welcher durch Personen oder Gruppen bei einer entsprechenden Aufgabe erreicht werden kann. [48] Unabhängig von der Definition ist das Heranziehen des geringsten Fehlers von Menschen für 𝜖ℎ𝑙𝑝 korrekt. Die niedrigste Fehlerquote, welche durch einen beliebigen Klassifikator erreicht werden kann, heißt Bayes Error und wird nachfolgend als 𝜖𝐵𝑎𝑦𝑒𝑠 bezeichnet. Es gilt hierbei immer 𝜖𝐵𝑎𝑦𝑒𝑠 ≤𝜖ℎ𝑙𝑝 . Da eine Schätzung von 𝜖ℎ𝑙𝑝 meist wesentlich leichter fällt und HLP bei vielen Aufgaben ohnehin sehr gut ist, kann 𝜖𝐵𝑎𝑦𝑒𝑠 = 𝜖ℎ𝑙𝑝 angenommen werden. Diese Annahme gilt jedoch lediglich unter der Voraussetzung, dass sowohl der menschliche als auch maschinelle Klassifikator unter identischem Informationsgehalt agieren. [19] 

## **2.5 Mathematische und Informationstechnische Grundlagen des Deep Learnings** 

Im Bereich der datengetriebenen Programmierung können oftmals bessere Ergebnisse durch die Erhöhung der Datenqualität oder -quantität erzielt werden. Mit einer zunehmenden Anzahl an Trainingsdaten steigt deshalb der Bedarf einer geeigneten Schreibweise und Datenstruktur zum Speichern und effizienten Manipulieren der Daten. Sowohl in der Mathematik als auch der Informatik bieten sich hierfür Tensoren an. [19] 

30 

2 Einführung in das Maschinelle Lernen 

## **2.5.1 Tensoren und Tensorbibliotheken** 

Matrizen spielen eine wichtige Rolle in der Linearen Algebra. Sie werden verwendet, um lineare Gleichungssysteme kompakt darzustellen, können aber genauso lineare Funktionsabbildungen darstellen. Eine wie in Formel 2.17 beschriebene reell-wertige (m, n)-Matrix ist ein Tupel bestehend aus (m, n)-Einträgen, welche rechteckig angeordnet sind und aus m Reihen sowie n Spalten besteht. [13] 

**==> picture [349 x 52] intentionally omitted <==**

Matrizen bezeichnen immer nur zweidimensionale Gitter aus Nummern. Da im Kontext von ML jedoch hochdimensionale Strukturen hilfreich sind, werden Tensoren statt Matrizen verwendet. Ein Tensor wird oft als eine verallgemeinerte Matrix betrachtet. Ein Tensor kann demnach eine 1D-Matrix, eine 3D-Matrix, sogar eine 0DMatrix oder höherdimensionale Struktur sein, welche dann schwer zu visualisieren ist. [13] Alle nachfolgenden Eigenschaften beziehen sich entsprechend auf Tensoren und nicht auf Matrizen. 

In der Informatik wird im Kontext von ML meist eine Tensorbibliothek verwendet, welche zahlreiche Funktionen zur Interaktion mit Tensoren liefert. Tensoren unterstützen eine Vielzahl verschiedener Operationen. Herkömmliche Operationen wie Addition, Subtraktion, Division und Multiplikation werden elementweise durchgeführt. Falls die Dimensionen zweier Operanden nicht übereinstimmen, ist die elementweise Funktion entweder nicht möglich oder einer der Operanden wird gebroadcastet. Broadcasting bezeichnet im Kontext von bekannten Tensorbibliotheken wie NumPy oder PyTorch die automatische Erweiterung der Tensorargumente, sodass diese die gleiche Größe haben. Broadcasting wird unterstützt wenn folgende zwei Regeln erfüllt sind: [33] 

- Jeder Tensor hat mindestens eine Dimension. 

- Bei der Iteration über die Größen der Dimensionen, beginnend mit der letzten Dimension, müssen die Größen der Dimensionen entweder gleich sein, eine von ihnen ist 1 oder eine von ihnen existiert nicht. 

Wenn zwei Tensoren 𝑋 und 𝑌 Broadcasting unterstützen, wird die resultierende Tensorgröße wie folgt berechnet: [33] 

- Wenn die Anzahl der Dimensionen von 𝑋 und 𝑌 nicht gleich ist, wird den Dimensionen des Tensors mit weniger Dimensionen eine 1 vorangestellt, damit diese gleich lang sind. 

- Danach ist die resultierende Dimensionsgröße für jede Dimensionsgröße das Maximum der Größen von 𝑋 und 𝑌 entlang dieser Dimension. 

31 

2 Einführung in das Maschinelle Lernen 

Neben elementweisen Operationen werden ebenfalls bei korrekter Dimensionalität der Operanden alle herkömmlichen Operationen wie beispielsweise MatrixMultiplikation unterstützt [33]. Für tiefgreifende Einblicke der zugrundeliegenden Mathematik, insbesondere Linearer Algebra, Optimierungsverfahren und Bayesscher Statistik wird auf [13] und [49] verwiesen. 

## **2.5.2 Automatische Differenzierung und Berechnungsgraphen** 

Aufgrund des hohen manuellen Aufwands und der Fehleranfälligkeit der partiellen Differenzierung in Bezug auf alle trainierbaren Parameter eines Netzwerkes werden im Zuge von Tensor- oder DL Frameworks Berechnungsgraphen bereitgestellt, sodass lediglich der Vorwärtspass des neuronalen Netzes konzipiert werden muss. Die automatische Differenzierung mit Bezug auf sämtliche trainierbare Parameter erfolgt über einen Berechnungsgraphen, welcher im Vorwärtspass erstellt wird. Ein Berechnungsgraph ist ein Graph, in dem jeder Knoten einer mathematischen Operation oder einer Variable entspricht. Sowohl Variablen können Werte in Operationen als auch Operationen deren Ergebnisse in andere Operationen weiterleiten. Abbildung 21 zeigt einen beispielhaften Berechnungsgraph mit Vorwärts- und Rückwärtspass. [33] 

**==> picture [454 x 314] intentionally omitted <==**

**----- Start of picture text -----**<br>
vorwärts w z<br>* MultBackward<br>y1 y2<br>log sin LogBackward SinBackward<br>a<br>* MultBackward<br>x1 x2 rückwärts<br>**----- End of picture text -----**<br>


_Abbildung 21: Beispielhafter Berechnungsgraph nach [33]_ 

32 

2 Einführung in das Maschinelle Lernen 

Ein neuronales Netz ist dementsprechend ein sehr komplexer Berechnungsgraph, bei dem jedes Neuron aus mehreren Knoten im Graph besteht, und die Ausgabe an eine gewisse Anzahl anderer Neuronen weiterleitet, bis der gesamte Graph durchlaufen ist. Aufgrund der Verkettung aller Variablen und Rechenoperationen im Berechnungsgraph kann dieser mittels rekursiver Anwendung der Kettenregel bezüglich jedes Inputparameters differenziert werden. Die Kettenregel für eine Variable ist durch Formel 2.18 beschrieben. 

**==> picture [326 x 13] intentionally omitted <==**

∘ Der Operator bezeichnet hierbei die Funktionskomposition. [50] Es gilt zu beachten, dass nicht alle Tensoroperationen differenzierbar sind, weshalb während der automatischen Konstruktion des Berechnungsgraphs auf die Verwendung ausschließlich differenzierbarer Funktionen geachtet werden muss. [33] 

## **2.6 Grundlagen des Geometrischen Deep Learnings** 

Zur Überwindung der Einschränkungen punkbasierter DL-Ansätze wurden oberflächen- und graphbasierte Methoden entwickelt, welche direkt auf der Maschenoberfläche operieren. Ansätze dieser Art fallen in den Bereich des GDL, welches ein grundlegendes Konzept für intrinsisches Lernen auf Oberflächen und Graphen darstellt. [51] Der Beginn ist auf „The Erlangen Programme“ des Mathematikers Felix Klein zurückzuführen und beschäftigt sich mit den Eigenschaften Nicht-Euklidischer Geometrie. Aufgrund der erfolgreichen Umsetzung von GDL auf Nicht-Euklidischen Geometrien folgt nun ein Einblick in die Topologielehre, Grundsätze des GDL und Eigenschaften nicht-euklidischer Geometrien sowie intrinsische Datenoperationen. [52] 

## **2.6.1 Topologie** 

Dieses Kapitel gibt einen Einblick in das sehr abstrakte Feld der Topologie und erläutert das für die nächsten Kapitel relevante Vokabular. Als Topologie wird die Domäne der Mathematik bezeichnet, welche sich mit Eigenschaften von Mengen beschäftigt, welche durch kontinuierliche Verformungen unbeeinflusst bleiben [53]. 

## **Homöomorphismen** 

Als topologisch äquivalent oder homöomorph werden zwei Topologieräume bezeichnet, welche ineinander überführt werden können. Topologische Eigenschaften können beispielhaft an der Verformung einer Kugel in einen Torus erklärt werden. Während die Verformung einer Kugel in ein Ellipsoid durch einfaches Dehnen möglich ist, kann eine Kugel nicht in ein Torus verformt werden, ohne diese 

33 

2 Einführung in das Maschinelle Lernen 

an einer Stelle auseinanderzuschneiden und wieder zu verknüpfen. [53] Zum Gegenbeweis topologischer Äquivalenz zweier Mannigfaltigkeiten werden üblicherweise topologische Invarianten verwendet. Dies sind Zahlen oder mathematische Objekte wie Gruppen, Matrizen, Polynome oder Vektorräume, die durch Homöomorphismen erhalten bleiben. Unterscheiden sich die Mannigfaltigkeiten in ihren Invarianten können diese dann zwangsläufig nicht homöomorph sein. Eine oftmals verwendete Invariante ist die Fundamentalgruppe einer Mannigfaltigkeit. Dabei stehen verschiedene Elemente der Fundamentalgruppe für unterschiedliche Möglichkeiten zum Zeichnen eines kontinuierlichen geschlossenen Pfades. Alle kontinuierlich ineinander verformbaren Schleifen werden als gleichwertig angesehen. Demnach ist die Anzahl der Schleifen eine Maß für die Anzahl der Löcher einer Mannigfaltigkeit. [54] 

## **Mannigfaltigkeiten** 

Die Topologie handelt im Wesentlichen von Mannigfaltigkeiten – teilweise auch als Vielfältigkeit bezeichnet. Grundlegend sind Mannigfaltigkeiten durch Kurven oder Oberflächen beschrieben, können jedoch beliebig viele Dimensionen aufweisen. Jede Mannigfaltigkeit wird über eine nichtnegative Zahl charakterisiert, welche - Auskunft über die Anzahl unabhängiger Parameter gibt. Der Prototyp einer dimensionalen Mannigfaltigkeit ist der -dimensionale euklidische Raum ℝ[𝑛] . In diesem ist die Beschreibung jedes Punktes über ein -Tupel aus reellen Zahlen möglich. Zur Charakterisierung einer -dimensionalen Mannigfaltigkeit wird diese als -Mannigfaltigkeit bezeichnet. 0-Mannigfaltigkeiten beschreiben lediglich eine diskrete Menge, also Punkte. 1-Mannigfaltigkeiten beschreiben Linien, Kurven oder beliebige stetige Funktionen. Raumkurven zählen ebenfalls zu 1-Mannigfaltigkeiten. Entscheidend in jedem dieser Beispiele ist die Möglichkeit einen eindeutigen Punkt mittels einer einzigen reellen Zahl zu spezifizieren. Zu den 2-Mannigfaltigkeiten zählen folglich Flächen. Die häufigsten Anwendungen sind Ebenen, sphärische Oberflächen, sowie die Flächen von Zylindern, Ellipsoiden, Paraboloiden und Hyperboloiden. 3-Mannigfaltigkeiten sind die letzten visualisierbaren Mannigfaltigkeiten in dreidimensionalem euklidischem Raum. Die mathematische Konstruktion höherdimensionaler Mannigfaltigkeiten ist jedoch typisch. [54] Abbildung 22 veranschaulicht die Verformung verschiedener Mannigfaltigkeiten. Mit blau gestrichelten Linien werden hierbei kontinuierlich geschlossene Schnitte symbolisiert. Die Kugel ist nach dem Schnitt topologisch homöomorph zu einer Scheibe. Dies zeigt auch, dass sich die Dimension zwischen nicht-homöomorphen Topologien nicht zwangsläufig unterscheiden muss. Durch das erneute Schneiden einer Scheibe entsteht ein Zylinder. Nach der Projektion der Punkte der inneren Grenze auf die äußere kann ein Torus geschaffen werden, welcher wiederum durch 

34 

2 Einführung in das Maschinelle Lernen 

erneutes Schneiden einen Zylinder bildet. Bei allen in Abbildung 22 dargestellten Objekten handelt es sich um 2-Mannigfaltigkeiten. [53] 

**==> picture [406 x 128] intentionally omitted <==**

**----- Start of picture text -----**<br>
Kugel Scheibe Zylinder Torus Zylinder<br>“<br>¥ \ - N (CEaa y - “ ieVy<br>a N . ><br>I " ley<br>\ ee fxr<br>N\ y)<br>Schnitt Schnitt Verknüpfung Schnitt<br>**----- End of picture text -----**<br>


_Abbildung 22: Topologische Verformungen verschiedener Mannigfaltigkeiten nach [53] und [54]_ 

Hierbei wird auch deutlich, dass zur Spezifizierung eines Punktes neben dem kartesischen Koordinatensystem auch andere Koordinatensysteme hilfreich sind wie beispielsweise Polarkoordinaten. 

## **2.6.2 Einordnung Geometrischen Deep Learnings** 

Unter GDL werden Methoden zur Generalisierung tiefer neuronaler Netze für nichteuklidische Geometrie verstanden. In diesen Bereich zählen unter anderem Gitter, Gruppen, Graphen, Geodäten und Eichungen. Den englischen Übersetzungen „ **G** rids, **G** roups, **G** raphs, **G** eodesics, **G** auges” zufolge werden die Kategorien in der Literatur auch als die „5G“ bezeichnet. Unter „Gruppen“ werden in diesem Zusammenhang globale Symmetrietransformationen im homogenen Raum, unter „Geodäten“ metrische Strukturen und Mannigfaltigkeiten und unter „Eichungen“ lokale Bezugsrahmen, die auf Tangentenbündeln definiert sind, verstanden. Abbildung 23 verbildlicht die 5 Kategorien. [52] 

35 

2 Einführung in das Maschinelle Lernen 

**==> picture [384 x 23] intentionally omitted <==**

**----- Start of picture text -----**<br>
Geodäten &<br>Gitter Gruppen Graphen<br>Eichungen<br>**----- End of picture text -----**<br>


_Abbildung 23: Die 5G des GDL: "Grids, Groups, Graphs, Geodesics & Gauges" nach [52]_ 

## **Riemannsche Mannigfaltigkeiten** 

Die im Rahmen dieser Arbeit verwendeten Daten lassen sich den letzten beiden Kategorien zuordnen, weshalb auf diese nun genauer eingegangen wird. Mannigfaltigkeiten verkörpern grundsätzlich zwei Arten der Invarianz. Hierzu zählen zum einen Transformationen, die die metrische Struktur erhalten und zum anderen lokale Bezugsrahmenwechsel. Während es für viele Datenstrukturen bereits etablierte Ansätze des ML gibt, ist das Lernen basierend auf Mannigfaltigkeiten ein eher neues Konzept, trotz der weiten Verbreitung dieser in der Physik und Computergrafik. Insbesondere im Bereich der Computergrafik sind Mannigfaltigkeiten ein gängiges mathematisches Modell für 3D-Formen. Der Begriff „3D“ ist hierbei irreführend und bezieht sich auf den Einbettungsraum. Die Formen selbst sind 2- Mannigfaltigkeiten, da diese lediglich die Grenzfläche bzw. Oberfläche eines 3DObjekts darstellen. [52] Mannigfaltigkeiten fallen in den Bereich variierender Domänen und sind daher Graphen sehr ähnlich, jedoch gegenüber Domänenverformungen invariant. Diese Eigenschaft ist auch unter geometrischer Stabilität bekannt und beruht auf der Differentialgeometrie. [51] Mannigfaltigkeiten können als eine glatte mehrdimensionale gekrümmte Oberfläche interpretiert werden, die lokal euklidisch ist, d.h. jede kleine Nachbarschaft um einen beliebigen Punkt zu einer Nachbarschaft von ℝ[𝑠] deformiert werden kann. Dies impliziert eine 𝑠 - dimensionale Mannigfaltigkeit und ermöglicht die Approximation der Mannigfaltigkeit um den Punkt 𝑢 durch den lokalen Tangentenraum 𝑇𝑢𝛺 . [52] Als Beispiel einer zweidimensionalen Mannigfaltigkeit bietet sich die Kugeloberfläche an, da bei einer ausreichenden Vergrößerung die Kugeloberfläche planar erscheint. Dieses Prinzip zeigt auch Abbildung 24. Ein Tangentenvektor, der als 𝑋∈𝑇𝑢𝛺 bezeichnet wird, kann als lokale Verschiebung um Punkt 𝑢 betrachtet werden. Zur Messung der Längen der Tangentenvektoren und deren Winkeln muss der Tangentenraum mit einer zusätzlichen Struktur ausgestattet werden, ausgedrückt als positiv bilineare Funktion 

36 

2 Einführung in das Maschinelle Lernen 

𝑔𝑢 ∶ 𝑇𝑢𝛺× 𝑇𝑢𝛺→ℝ , die gleichmäßig von 𝑢 abhängt. Eine solche Funktion wird als Riemannsche Metrik bezeichnet und kann als inneres Produkt des Tangentenraums 〈𝑋, 𝑌〉𝑢 = 𝑔𝑢〈𝑋, 𝑌〉 betrachtet werden, welches den Winkel zwischen zwei beliebigen Tangentenvektoren 𝑋, 𝑌∈𝑇𝑢𝛺 ausdrückt. Zudem induziert die Metrik auch die Norm 1⁄2 ‖𝑋‖𝑢 = 𝑔𝑢 (𝑋, 𝑋) , die die lokale Messung der Längen der Vektoren erlaubt. Eine mit einer Riemannschen Metrik versehene Mannigfaltigkeit wird als Riemannsche Mannigfaltigkeit bezeichnet. Alle durch die Metrik vollständig ausdrückbaren Eigenschaften werden als intrinsisch bezeichnet. [52][51] 

**==> picture [366 x 116] intentionally omitted <==**

**----- Start of picture text -----**<br>
𝑥 𝑇𝑢𝕊2 𝑇𝑢𝕊2<br>𝑥, 𝑦 𝑢<br>𝑥<br>𝑢 𝑦 𝑢 𝑢<br>𝑥<br>𝑇𝑢𝕊2 𝑑(𝑢, 𝑣)<br>𝑒𝑥𝑝𝑢𝑥<br>𝛾(𝑡) \ 𝑣 𝑢→ 𝑥<br>hy XCD<br>𝑇 𝕊2<br>𝕊2<br>**----- End of picture text -----**<br>


_Abbildung 24: Grundbegriffe der Riemannschen Geometrie am Beispiel der zweidimensionalen Kugel nach [52]_ 

Der Tangentenraum an die Sphäre ist gegeben als 𝑇𝑢𝕊[2] = {𝑥∈ℝ[3] ∶ 𝑥[𝑇] 𝑢= 0} und ist eine 2D-Ebene – also eine zweidimensionale Mannigfaltigkeit. Die Riemannsche Metrik ist einfach das auf die Tangentialebene beschränkte euklidische innere Produkt, 〈𝑥, 𝑦〉𝑢 = 𝑥[𝑇] 𝑦 für jedes 𝑥, 𝑥∈𝑇𝑢𝕊[2] . Die Exponentialkarte ist gegeben durch 𝑠𝑖𝑛(‖𝑥‖) 𝑒𝑥𝑝𝑢(𝑥) = 𝑐𝑜𝑠(‖𝑥‖) 𝑢+ 𝑥 für 𝑥∈𝑇𝑢𝕊[2] . Geodäten sind hier die großen Bögen ‖𝑥‖ der Länge 𝑑(𝑢, 𝑣) = 𝑐𝑜𝑠[−1] (𝑢[𝑇] 𝑣) . [52][51] 

## **Geodäten** 

Betrachtet man eine glatte Kurve 𝛾∶ [0, 𝑇] →Ω auf einer Mannigfaltigkeit mit den Endpunkten 𝑢= 𝛾(0) und 𝑢= 𝛾(𝑇) so ist die Ableitung der Kurve im Punkt 𝑡 ein Tangentenvektor 𝛾[′] (𝑡) ∈𝑇𝛾(𝑡)Ω , der als Geschwindigkeitsvektor bezeichnet wird. Unter allen Kurven, welche 𝑢 und 𝑣 miteinander verbinden, wird als Geodäte diejenige Linie bezeichnet, deren Länge minimal ist. Entsprechend wird ein 𝛾 gesucht, welches das Längenfunktional aus Formel 2.19 minimiert. [52] 

**==> picture [381 x 31] intentionally omitted <==**

Mittels Geodäten ist der Paralleltransport von Tangentenvektoren auf Mannigfaltigkeiten, die Erstellung lokaler intrinsischer Exponentialkarten der 

37 

2 Einführung in das Maschinelle Lernen 

Mannigfaltigkeit zum Tangentenraum und die Definition geodätischer Metriken möglich. Diese Eigenschaften ermöglichen die Konstruktion faltungsähnlicher Operationen, indem ein Filter lokal im Tangentenraum angewendet wird. Ein Problem im Umgang mit Mannigfaltigkeiten besteht darin, dass die Addition oder Subtraktion zweier Punkte 𝑢, 𝑣 ∈Ω nicht direkt möglich ist. [51] Ein identisches Problem findet sich im Vergleich von Tangentenvektoren an verschiedenen Punkten. Trotz der identischen Dimension gehören diese zwei Tangenten unterschiedlichen Tangentenräumen an und sind daher nicht direkt vergleichbar. Geodäten bieten hierfür ein Verfahren zur Bewegung von Vektoren von einem Punkt zu einem anderen Punkt. Wenn die Geodäte 𝛾 die Punkte 𝑢= (0) und 𝑣= (𝑇) verbindet, kann eine neue Menge von Tangentenvektoren 𝑋(𝑡) ∈𝑇𝛾(𝑡)𝛺 entlang der Geodäte definiert werden (Formel 2.20), sodass die Länge von 𝑋(𝑡) (Formel 2.21) und der Winkel zwischen ihm und dem Geschwindigkeitsvektor der Kurve konstant ist. [52] 

**==> picture [362 x 16] intentionally omitted <==**

**==> picture [325 x 14] intentionally omitted <==**

Das Resultat hiervon ist ein eindeutiger Vektor 𝑋(𝑡) ∈𝑇𝑢𝛺 am Endpunkt 𝑢 . Die als 𝛤𝑢→ (𝑋) = 𝑋(𝑇) definierte Exponentialkarte 𝛤𝑢→ (𝑋) ∶ 𝑇𝑢𝛺→𝑇 𝛺 wird in der obigen Schreibweise als Paralleltransport oder Verbindung bezeichnet. Der Begriff „Verbindung” bezieht sich hierbei auf die Tangentenräume 𝑇𝑢𝛺 und 𝑇 𝛺 . Aufgrund der Winkel- und Längenerhaltungsbedingungen läuft der Paralleltransport nur auf eine Drehung des Vektors hinaus, sodass er einem Element der speziellen orthogonalen Strukturgruppe des Tangentenbündels zugeordnet werden kann. [51] Abbildung 25 veranschaulicht den Paralleltransport eines Tangentenvektors von 𝐴 nach 𝐶 . [52] 

**==> picture [152 x 108] intentionally omitted <==**

**----- Start of picture text -----**<br>
𝐴<br>oe<br>~<br>™<br>»<br>~~ 𝐶<br>𝐵<br>**----- End of picture text -----**<br>


_Abbildung 25: Paralleltransport des Tangentenvektors A nach C nach [52]_ 

38 

2 Einführung in das Maschinelle Lernen 

Daraus wird ersichtlich, dass der euklidische Transport des Vektors A nach C auf der Kugel sinnlos ist, da die resultierenden Vektoren (rot) nicht in der Tangentialebene liegen. Der Paralleltransport von 𝐴 nach 𝐶 (blau) rotiert den Vektor entlang des Pfades. Es wird zudem deutlich, dass das Ergebnis pfadabhängig ist, da 𝐵𝐶 ein anderes Resultat als 𝐴𝐵𝐶 erzeugt. [52] 

Dieses Kapitel beschreibt lediglich die absoluten Grundlagen. GDL im Kontext von Mannigfaltigkeiten beschäftigt sich insbesondere noch mit Skalar- und Vektorfeldern, intrinsischen Gradienten, Exponentialkarten, geodätischen Abständen, Isometrien, intrinsischen Symmetrien, Fourier-Analysen sowie spektralen und räumlichen Faltungen. Hierfür wird auf [52] und [51] verwiesen. 

## **2.6.3 Intrinsische Datenoperationen** 

Das zunehmende Interesse an der Anwendung von ML auf geometrischen Daten hat dazu geführt bisher auf 2D-Daten erfolgreiche Prinzipien auf geometrischen 3DDaten anzuwenden. Der entscheidende Nachteil dieses Ansatzes, liegt in der Behandlung der geometrischen Daten als euklidische Strukturen. Dies gilt es aus zwei Gründen zu vermeiden. Erstens können durch die euklidische Darstellung von komplexen 3D-Objekten wie Tiefenbildern oder Voxel, wesentliche Teile des Objekts oder feine Details verloren gehen oder die topologische Struktur sogar zerstört werden. Zweitens sind euklidische Darstellungen nicht intrinsisch und ändern sich abhängig von der Pose oder Verformung des Objekts. Das Erreichen der Invarianz gegenüber Formveränderungen ist eine häufige Anforderung in der Datenverarbeitung und erfordert aufgrund der hohen Anzahl an Freiheitsgraden bei der Beschreibung nicht starrer Verformungen sehr komplexe Modelle sowie große Trainingsmengen. Abbildung 26 zeigt, dass beim Anwenden eines klassischen CNNs auf ein als euklidisch angesehenes Objekt der Filter zwar korrekt funktioniert, jedoch nicht verformungsinvariant auf der wirklichen, zugrundeliegenden, nicht-euklidischen Datenstruktur ist. Der Filter eines geometrischen CNNs operiert dagegen direkt auf den Daten, statt sich auf die Datenstruktur zu projizieren. Da beim geometrischen CNN keine Abstraktion stattfindet und der Filter intrinsisch angewendet wird, ist dieser auch invariant gegenüber Verformungen. [51] 

2 Einführung in das Maschinelle Lernen 

**==> picture [425 x 471] intentionally omitted <==**

**----- Start of picture text -----**<br>
2 Einführung in das Maschinelle Lernen  39<br>Filter<br>Filter<br>a ae<br>A A<br>Euklidisches Geometrisches<br>CNN CNN<br>**----- End of picture text -----**<br>


_Abbildung 26: Veranschaulichung des Unterschieds zwischen klassischem und geometrischem CNN nach [51] und [52]_ 

Insbesondere im Bereich der Computergrafik ist die Arbeit mit geometrischen Formen geläufig. Diese werden üblicherweise als Riemannsche Mannigfaltigkeiten, wie in Kapitel 2.6.2 beschrieben, modelliert und als Netze diskretisiert. Zahlreiche Studien haben sich mit der Entwicklung lokaler und globaler Merkmale befasst, um beispielsweise Ähnlichkeiten oder Korrespondenzen zwischen verformbaren Formen mit garantierter Invarianz zu Isometrien herzustellen. [51] 

40 

3 Bestehende Ansätze und Ableitung des Forschungsbedarfs 

## **3 Bestehende Ansätze und Ableitung des Forschungsbedarfs** 

Nach [1] ist die Verdrahtung der Schaltschrankkomponenten für bis zu 49 % der Durchlaufzeit verantwortlich. Der Grund hierfür ist die manuelle Ausführung dieser Aufgabe. Zur Reduktion der Verdrahtungszeit bestehen in der Literatur mehrere Ansätze. Diese werden nachfolgend zuerst aus der Perspektive der Schaltschrankmontage sowie den zugrundeliegenden ML-Prinzipen erläutert, um abschließend den Forschungsbedarf dieser Arbeit abzuleiten. 

## **3.1 Bestehende Ansätze zur Automatisierung in der Verdrahtung von Schaltschränken** 

Einige Hersteller sehen die Zukunft des Schaltanlagenbaus in der digitalen Fertigung, bei der die Verdrahtung automatisiert und komplett ohne Unterstützung stattfindet. Hierfür bestehen bisher nur sehr wenige Ansätze, welche sich zudem aus wirtschaftlichen Gründen auf die Automatisierung von Serienprodukten beziehen. Aufgrund der sehr hohen Variantenanzahl an Komponenten für Steuer- und Schaltanlagen besteht weder ein Konzept zur automatischen Verdrahtung dieser noch ein Verfahren zur Analyse der Komponenten zur Extraktion wichtiger Informationen. Bisherige Ansätze verfolgen das automatisierte Erkennen von Schaltern in der Schaltanlage in der Fertigung basierend auf RGB-Bildern sowie Template-Matching-Algorithmen zur Identifikation und Orientierungsbestimmung von Komponenten im Schaltschrank mittels Bildern und Punktwolken. [6] Der Ursprung aller Ansätze liegt dennoch in Daten aus der tatsächlichen Fertigung, anstelle der rohen CAD-Modelle, welche selbst viele Informationen über Anschlüsse enthalten und verfolgen meist das Ziel die reale Fertigung direkt zu steuern. [3] Auffällig ist zudem, dass ML sehr wenig Anwendung findet und verwendete Methoden nicht dem aktuellen Stand der Technik entsprechen. Ergänzend zum Forschungsanliegen dieser Arbeit liegen auch schon Verfahren zur Drahtspitzenerkennung vor, welche ebenfalls auf die korrekten Positionen der Kontaktierungen angewiesen sind. Dies unterstreicht die Bedeutung dieser Arbeit. [2] 

Aus der aufgeführten Literatur wird deutlich, dass zwar an der Automatisierung der Schaltschrankverdrahtung geforscht wird, sich die Forschung jedoch stark auf die Prozessoptimierung und herkömmliche Automatisierungsverfahren bezieht. ML wird nur wenig in der Fertigung von Schaltschränken und bislang gar nicht in der Verdrahtung der Komponenten angewendet. [1] 

41 

3 Bestehende Ansätze und Ableitung des Forschungsbedarfs 

## **3.2 Bestehende Ansätze zur Analyse von Dreiecksnetzen mittels Maschineller Lernverfahren** 

Die Anwendung von ML-Modellen auf dreidimensionale Daten wird mit dem stetig zunehmenden Grafikspeicher immer zugänglicher. Die Klassifizierung von Punktwolken oder Dreiecksnetzen erfolgt mittlerweile mit einer sehr hohen Genauigkeit. Hierfür werden in der Literatur zahlreiche verschiedene Ansätze wie Dynamic Graph CNNs, Point CNNs und Diffusionsnetze untersucht. [55][56] Für eine gute Vergleichbarkeit der Ergebnisse werden dabei typischerweise standardisierte Datensätze verwendet. Zu diesen zählen unter anderem: 

- RNA-Surface Segmentation Dataset [57] 

- ShapNet [58] 

- SHREC [59] 

Reale Anwendungen finden sich bisher vor allem im Bereich der Medizin und Biologie, jedoch nicht in den Ingenieurswissenschaften oder der Fertigungsindustrie [57][60]. Unabhängig von den erprobten Architekturen der neuronalen Netze gibt es bestehende Ansätze in den Bereichen Multi-View-2D-Projektionen, volumetrische Transformationen, Graphrepräsentationen und intrinsischem Lernen auf Mannigfaltigkeiten. [61] Bestehende Ansätze klassifizieren bis zu 91,5 % und segmentieren bis zu 92,3 % aller Datenpunkte korrekt. Erfolgreiche Netzwerkarchitekturen sind in diesem Kontext sowohl das DiffusionNet sowie MeshCNN. [56][61] Beide Verfahren bieten sich demnach zur Erprobung eines neuartigen Datensatzes an. Während die Ergebnisse auf Benchmark-Datensätzen sehr gut ausfallen, liegen die Ergebnisse in realen Anwendungen wie beispielsweise MedMeshCNN deutlich geringer. In der Segmentierung von sehr einfachen Aneurysmen wird beispielsweise ein durchschnittlicher Jaccard-Index von 63,24 % erzielt. [60] 

## **3.3 Ableitung des Forschungsbedarfs** 

Da bis zu 27 % der Unternehmen der Schaltschrankindustrie die Schaltschränke vorkonfektioniert, d.h. mit allen Durchbrüchen und Aussparungen versehen, beziehen, liegt der Anteil der mechanischen Bestückung und Verdrahtung bei ca. 75 % der Bearbeitungszeit. Im Gegensatz zu den anderen bereits halb- oder vollautomatisierten Fertigungsschritten profitiert die mechanische Bestückung und Verdrahtung durch das Potenzial der Digitalisierung und Automatisierung mittels moderner Verfahren des ML. Da die Funktionsweise eines Schaltschranks erst durch die finale elektrische Verdrahtung entsteht, kommt dieser Aufgabe eine hohe Bedeutung zu. Für die automatisierte Verdrahtung werden Informationen über die Art der Quelle und des Ziels sowie deren Positionen benötigt. [1] Aufgrund der digitalen Konstruktion des Schaltschranks und oftmals der Verwendung eines Digitalen 

42 

3 Bestehende Ansätze und Ableitung des Forschungsbedarfs 

Zwillings sind diese Informationen nicht nur während der Fertigung der realen Objekte, sondern auch für das digitale Abbild relevant. Der Grundstein beide Fälle gemeinsam abzudecken, besteht in der Identifikation und Positionsbestimmung der relevanten Merkmale einer Komponente. Aktuelle Ansätze basierend auf DL zeigen vielversprechende Ergebnisse auf Datensätzen wie SHREC und ShapeNet. [59][58] Entgegen diesen Datensätzen ist die Handhabung eigens erhobener Daten meist schwieriger [60]. Hinzu kommt, dass die Komplexität der Geometrie von Schaltschrankkomponenten um ein Vielfaches höher ist als die der Objekte in den erwähnten Datensätzen. 

Ausgehend von diesen Aspekten bietet es sich an den aktuellen Stand der Technik im Bereich der Teilsegmentierung von Dreiecksnetzen, sowie Clustering-Algorithmen und zweidimensionaler Bilddatenverarbeitung am Beispiel von Schaltschrankkomponenten zu erproben. Abgesehen von der branchenspezifischen Anwendung, bietet das Forschungsanliegen zudem die Möglichkeit ein allgemeines Konzept und Fazit zu erarbeiten, um Montagebaugruppen auf Basis der Geometrie zu analysieren. 

43 

4 Angewandte Methodik 

## **4 Angewandte Methodik** 

Die Arbeit orientiert sich an der DMME und bildet insbesondere die ersten sieben Schritte dieser Methodik ab. DMME beschreibt einen Leitfaden zur Umsetzung von Data Mining Aufgaben im industriellen Umfeld. Im Weiteren werden die einzelnen Schritte der DMME erfasst und die Methodik von weiteren Ansätzen abgegrenzt. Implementierungstechnische Überlegungen bezüglich der Softwareentwicklung werden in Kapitel 4.2 genauer erläutert. 

## **4.1 Data Mining Methodology for Engineering Applications** 

Durch neue aufkommende Technologien des Industrial Internet of Things, CyberPhysical Production Systems und technischen Möglichkeiten der Industrie 4.0 steigt die Anzahl potenziell nützlicher Datenmengen für manufakturbezogene Prozesse enorm an. Die zur Überwachung der Produktionsprozesse nötigen Systeme sammeln so viel Daten in Echtzeit, dass die kontinuierliche Verwertung dieser Daten einige Komplikationen hervorruft. Ebenso steigt die Komplexität der Datenverarbeitung weiter an, wodurch domänenspezifisches Expertenwissen gefragter wird. [62] Über die Jahre sind deshalb viele Data Mining (DM) Konzepte entstanden, welche sich meist aus dem Knowledge Discovery in Databases (KDD) Konzept ableiten lassen. Mittlerweile spielt dieses jedoch eine untergeordnete Rolle, weshalb die “Sample, Explore, Modify, Model and Assess“ (SEMMA) Methodik und der Cross-Industry Standard Process for Data Mining (CRISP-DM) als industrieller Leitfaden dienen, um die aufkommenden Herausforderungen zu bewältigen [63]. Ersterer fokussiert sich dabei auf die technische Integration von DM-Tools, wird allerdings durch die mangelnde Berücksichtigung organisatorischer Prozesse zunehmend irrelevanter. [64] Auch trotz der weiten Akzeptanz von CRISP-DM im industriellen Umfeld beachtet dieser dagegen nicht die fachbezogenen Komplikationen bei der Datenbeschaffung und -verarbeitung sowie das für den technischen Kontext relevante Wissen. Das technische Verständnis der zu analysierenden Objekte, sowie die technische Realisierung und Implementierung sind daher kein Bestandteil des CRISP-DM. [62] Zur Eliminierung dieser fehlenden Aspekte wurde eine ganzheitlichere Betrachtung über die Datenanalyse im Produktionsumfeld geschaffen und in der DMME festgehalten. Die Anwendung von DMME ist deshalb als unterstützendes Konzept für Optimierungen über die gesamte Wertschöpfungskette hinweg geeignet, um datengetriebene Entwicklungsziele iterativ zu erreichen. Abbildung 27 zeigt farblich hinterlegt durch welche Prozessschritte CRISP-DM konkret ergänzt wurde und bildet DMME ganzheitlich ab. [63] 

44 

4 Angewandte Methodik 

**==> picture [454 x 272] intentionally omitted <==**

**----- Start of picture text -----**<br>
Geschäfts-<br>verständnis<br>Technisches<br>Bereitstellung<br>Verständnis<br>Technische Technische<br>Implementierung Realisierung<br>Daten-<br>Evaluierung<br>verständnis<br>Daten-<br>Modellierung<br>aufbereitung<br>**----- End of picture text -----**<br>


_Abbildung 27: DMME als holistische Erweiterung des CRISP-DM nach [62]_ 

Im Vergleich zu CRISP-DM wird DMME durch die drei Schritte „Technisches Verständnis”, „Technische Realisierung“ und „Technische Implementierung“ erweitert. [62] Zu Beginn der DMME-Methodik muss zunächst das Geschäftsverständnis aufgebaut werden, um Ziele und Anforderungen der Arbeitsaufgabe aus der Unternehmensperspektive zu verstehen. Danach erfolgt eine Umwandlung des erworbenen Wissens in eine DM-Problemdefinition und einen vorläufigen Plan zur Erreichung des Ziels. [65] Vor der Umsetzung datengetriebener Projekte muss zunächst ein technisches Verständnis zur Bewältigung der Aufgaben geschaffen werden. Dies ist die Grundvoraussetzung für eine auswertbare Datenbasis. [63] Die Ziele der zweiten Phase bestehen in der Umwandlung geschäftlicher in technische Zielgrößen, zur Gewährleistung der Messbarkeit dieser. [66] Darüber hinaus umfasst sie die Erhebung prozessbezogener Wechselwirkungen, die Konzeptualisierung einer DM-Idee sowie die Entwicklung eines Test- und Versuchsplans. [62] In der folgenden technischen Realisierung erfolgt die Prototypisierung der zuvor erarbeiteten Datenerfassungskonzepte. Ein optimales Ergebnis liegt vor, wenn die generierten Daten den Anforderungen gerecht werden. Wichtige Faktoren sind beispielsweise die Datenqualität und der Informationsgehalt, um nachfolgende Datenanalysen durchzuführen, Thesen aufzustellen und diese zu belegen. [63] Das Datenverständnis wird durch die explorative Datenauswertung, die Bewertung der Datenqualität sowie Visualisierungen der Daten erreicht. Die folgende Datenaufbereitung umfasst alle 

45 

4 Angewandte Methodik 

Aktivitäten, die zur Überführung der Rohdaten in einen für die Modellierung endgültigen Datensatz erforderlich sind. Hierzu zählen unter anderem die Datenaufbereitung und -transformation. [65][66] Diese befasst sich mit der Auswahl und Erstellung der Modelle. Üblicherweise werden mehrere verschiedene Modellarchitekturen implementiert, wodurch oftmals auch dafür spezifische Anforderungen in der Datenaufbereitung entstehen. [66] Vor der Implementierung des besten Modells in einer Produktivumgebung werden die Ergebnisse zunächst zusammen mit Experten bewertet und interpretiert, um abschließend die Erreichung der zu Beginn gesetzten Geschäftsziele zu beurteilen. [65] Die technische Implementierung und Bereitstellung ist lediglich für den Produktivbetrieb notwendig und nicht Teil dieser Arbeit. 

## **4.2 Software-Entwicklung** 

Da sämtlicher in Anhang D – Digitaler Anhang beigefügter Code fast ausschließlich auf Python basiert, liegt der Fokus der Entwicklungsmethodik auf Python und dessen Umgebung selbst. Die Entwicklung hat entsprechend unter Berücksichtigung und Evaluierung der Best Practices der Python Entwicklung stattgefunden. Die Methodik basiert deshalb auf Programmierrichtlinien, der Entwicklung performanter Programme und hardwarespezifischen Implikationen im Kontext von KI und ML. 

## **4.2.1 Programmierrichtlinien** 

Die meisten Fehler in der Software-Entwicklung fallen in bekannte Kategorien aufgrund der Neigung des Menschen unbeabsichtigt repetitive Fehler zu generieren. Aufgrund dieser Vorhersehbarkeit wird die statische Code-Analyse verwendet, um Fehler während der Entwicklung oder zeitnah zu identifizieren. Während der statischen Code-Analyse wird die Software nicht ausgeführt, sondern lediglich auf bestimmte Muster geprüft. [67] 

Die Entwicklung im Rahmen dieser Arbeit hat unter Berücksichtigung des Python Enhancement Proposals (PEP) stattgefunden. Hierbei lag der Fokus insbesondere auf der Einhaltung folgender PEP-Standards: [68] 

- PEP7: Style Guide for C Code 

- PEP8: Style Guide for Python Code 

- PEP257: Docstring Conventions 

Zur Einhaltung der PEP-Standards wurde das Package autopep8 verwendet und in die statische Codeanalyse eingebaut [69]. Die hierfür verwendete Konfigurationsdatei ist in Anhang A – Autopep8 Konfigurationsdatei beigefügt. Die PEP8-Konvention wurde jedoch nicht erzwungen, um die etablierte stark mathematische Notation in ML-Code im Rahmen dieser Arbeit sinnvoll anzuwenden. Da die Wartbarkeit und Reproduzierbarkeit von ML-Projekten stark von deren Softwareabhängigkeiten 

46 

4 Angewandte Methodik 

abhängen, sind voneinander unabhängige und eigenständige Packages entwickelt worden, sodass diese mittels Pip Installs Packages (pip) installiert werden können. [70] Die Softwareabhängigkeiten sind entsprechend in Anforderungs- oder Umgebungsdateien hinterlegt. [71] Für die entwickelten Software-Pakete wurde die softwareseitige Dokumentation mittels des Dokumentationsgenerierungstools Sphinx automatisch auf Basis der im Code beschriebenen Docstrings erstellt [72]. Während der lokalen Entwicklung wurde zur Vermeidung von Versionskonflikten mit virtuellen Umgebungen gearbeitet. [70] Hierfür wurde der Binärpaketmanager Conda verwendet, welcher insbesondere im Data Science Umfeld häufig Anwendung findet. [73] 

## **4.2.2 Gewährleistung hochperformanter Programme** 

Trotz des Trainings neuronaler Netze auf darauf ausgelegter Hardware, laufen insbesondere Großteile der Datenvorverarbeitung auf der Central Processing Unit (CPU) ab und sind dadurch in der Parallelität hardwaretechnisch stärker begrenzt. Zur Gewährleistung performanter Programme werden für diese Arbeit deshalb zwei Prinzipien verfolgt. 

## **Python C API** 

Trotz der Überlegenheit der Programmiersprache Python im Bereich ML und Data Science ist diese Sprache aufgrund des Python Interpreters äußerst langsam in der Ausführung im Vergleich zu kompilierten hardwarenahen Sprachen. Da Python selbst in C geschrieben wurde, bestehen standardmäßig Möglichkeiten zur Interaktion zu den Sprachen C und C++. Die Python C API stellt die allgemeine Schnittstelle zwischen Python, C und C++ her, sodass der Zugriff auf den Python Interpreter auf einer Vielzahl von Ebenen möglich ist. Der Verwendung der Python C API liegen zwei Argumente zugrunde. Zum einen können Module zur Erweiterung des Python Interpreters geschrieben werden und zum anderen ist die Einbettung von in Python geschriebenen Programmen in anderen Applikationen möglich. Für diese Arbeit ist lediglich ersterer Grund relevant. Als Leitlinie während der Softwareentwicklung wurden im Rahmen dieser Arbeit deshalb stark iterative Aufgaben teilweise in C Programme ausgelagert und über die Python C API in die eigentliche Hauptroutine in Python eingebettet. [74] Darüber hinaus wurde darauf geachtet primär Bibliotheken zu verwenden, welche in C oder C++ geschrieben sind und lediglich Python Bindings aufweisen oder zumindest vektorisiert sind. 

## **Vektorisierung** 

Zur effizienten Datenverarbeitung und -manipulation bieten sich Tensoren an. Diese sind insbesondere für Anwendungen des Maschinellen Sehens hilfreich, da Bilder, 

47 

4 Angewandte Methodik 

Punktwolken und Polygonnetze effizient in Tensoren dargestellt werden können. Die am weitesten verbreitete Bibliothek für wissenschaftliches Rechnen in Python ist NumPy. [75] Aufgrund der zentralen Stellung von NumPy im Python-Ökosystem wird es zunehmend zur Interoperabilitätsschicht zwischen verschiedenen ArrayBerechnungsbibliotheken, welche sich spezifischeren Aufgaben widmen. Aufgrund der starken Integration von NumPy in sämtliche Python-Pakete bildet es den Grundstein wissenschaftlicher Forschung in Python. NumPy vereinigt mehrere grundlegende Array-Konzepte: [76] 

- Array-Datenstruktur 

- Indizierung 

- Vektorisierung 

- Broadcasting 

- Reduktionsoperationen 

Das Array-Objekt stellt ein allgegenwertiges Objekt in NumPy dar, auf welchem oder mittels dessen alle Operationen ausgeführt werden können und wird nachfolgend nur noch als Array bezeichnet. Arrays können über Slices und Schrittweiten indiziert werden, wodurch eine View der Daten generiert wird. Der Vorteil dieser Operationen liegt in der enormen Effizienz, da Datenmanipulationen lediglich durch unterschiedliche Interpretationen des Random Access Memory (RAM) Layouts erfolgen, Daten jedoch nicht neu geschrieben werden müssen. Zusätzlich können Arrays über Masken indiziert werden, wodurch eine Kopie des indizierten Bereichs zurückgegeben wird. Vektorisierung meint hier das Anwenden einer Operation auf eine Gruppe von Elementen. Vektoroperationen sind in NumPy über mehrere Basic Linear Algebra Subprograms (BLAS) Backends implementiert. Diese sind für die jeweiligen CPUs optimiert und verfolgen das Ziel der Minimierung der Cache Misses, um die enorme Rechenleistung der Register auszunutzen. Reduktionsoperationen wirken entlang einer oder mehrerer Achsen des Arrays und verändern üblicherweise die Dimensionalität des Arrays. [76] Da PyTorch ein zu NumPy fast identisches Application Programming Interface (API) aufweist, können für die CPU ausgelegte Programme, welche starken Gebrauch großer Arrays machen, leicht für eine Graphics Processing Unit (GPU) angepasst werden. [33] 

## **4.2.3 Implikationen der Hardware** 

Aufgrund der hohen benötigten Rechenleistung im Kontext von ML, wird das Training neuronaler Netze auf GPUs durchgeführt, welche ein Vielfaches der Kerne von CPUs aufweisen. Aus dem grundlegend abweichenden Hardware-Aufbau von GPUs im Vergleich zu CPUs, bedarf es einer darauf ausgerichteten Programmiersprache. Diese werden dem General Purpose Computation on Graphics Processing Unit (GPGPU) Computing zugeordnet. [77] Abhängig von der konkreten GPU können verschiedene GPGPU-Programmiersprachen verwendet werden. Die am weitest 

48 

4 Angewandte Methodik 

verbreiteten Sprachen sind OpenCL und Compute Unified Device Architecture (CUDA). OpenCL ist gegensätzlich zu CUDA ein offener Standard für GPGPU, jedoch aufgrund der weiten Verbreitung von NVIDIA GPUs im Kontext von DL bisher weniger geeignet. [78] Für eine vergleichende Analyse zwischen OpenCL und CUDA wird auf [79] verwiesen. Da im Rahmen dieser Arbeit eine NVIDIA GeForce RTX 2080 Ti zur Verfügung steht, wird ergänzend CUDA als Hardwarebeschleuniger verwendet. Die Hardware- und Softwarespezifikationen der in dieser Arbeit verwendeten GPU sind in Tabelle 1 aufgelistet. 

_Tabelle 1: Hardware- und Softwarespezifikationen der NVIDIA GeForce RTX 2080 Ti_ 

|Spezifikation|Wert|
|---|---|
|Grafikkarte|NVIDIA GeForce RTX 2080 Ti|
|Grafikspeicher|11GB|
|CUDA-Treiber|470.57.02|
|NVCC|10.1.243|
|CUDA-Toolkit|11.4|



Da abgesehen vom Training der ML-Modelle ein Großteil auf der CPU berechnet wird und beim Arbeiten mit sehr großen Datenmengen Komplikationen hinsichtlich des RAM auftreten können, listet Tabelle 2 relevante Hardwareund Softwarespezifikationen zum verwendeten Personal Computer (PC) auf. 

_Tabelle 2: Hardware- und Softwarespezifikationen des verwendeten PCs_ 

|Spezifikation|Wert|
|---|---|
|OS|Ubuntu 20.04.03.LTS|
|RAM|32 GB|
|CPU|Intel Core i7-6700K|
|CPU-Kerne|4|
|L1-Cache|128 KB|
|L2-Cache|1024 KB|
|L3-Cache|8192 KB|



Da insbesondere neue Software-Releases dazu tendieren zuerst Linux Betriebssysteme (OS) zu unterstützen, wurde die gesamte Software auf Ubuntu 20.04.03 entwickelt und getestet. Gegen eine Nutzung auf Windows oder MacOS 

49 

4 Angewandte Methodik 

spricht grundsätzlich nichts, wobei die Funktionalität nicht gewährleistet ist. Für weitere Forschungszwecke sollte Linux verwendet werden, da auch im Rahmen dieser Arbeit Bibliotheken erprobt wurden, welche bisher nicht oder nur teilweise Windows und MacOS unterstützen. Unter Berücksichtigung dieser Hardwarespezifikationen werden in Kapitel 5 die Daten vorverarbeitet und neuronale Netze implementiert. 

50 

5 Datenaggregation, Modellierung und Technische Implementierung 

## **5 Datenaggregation, Modellierung und Technische Implementierung** 

Ausgehend von der im letzten Kapitel vorgestellten Methodik folgt nun die Datenvorverarbeitung, Modellierung und Technische Implementierung der neuronalen Netze. Da alle Daten im Rahmen der Arbeit erhoben wurden, wird zunächst die Datenaggregation erläutert. Anschließend werden die Labels der Daten betrachtet und darauf aufbauend Deep Learning Modelle implementiert. Das Kapitel endet mit technischen Möglichkeiten zur Optimierung der implementierten Modelle. 

## **5.1 Datenerhebung** 

Alle erläuterten Ansätze dieser Arbeit bauen auf einem Datensatz bestehend aus 234 gelabelten Dreiecksnetzen von Schaltschrankkomponenten auf. Dieses Kapitel erläutert die Art und Weise der Datenerhebung sowie die Datenaufbereitung für die folgenden ML-Modelle. 

## **5.1.1 Datenaggregation** 

Problemlösungen basierend auf ML bieten sich nur an, wenn eine Vielzahl qualitativ hochwertiger Daten vorhanden ist. Zum Erlernen geometrischer Features von Schaltschrankkomponenten werden entsprechend viele dieser benötigt. Mögliche Vorgehensweisen zum Erfassen der Schaltschrankkomponenten bestehen im 3DScannen realer Objekte, dem eigenständigen Modellieren der 3D-CAD-Modelle und dem strukturierten Scrapen des World Wide Webs. Aufgrund der oftmals schlechten Rekonstruktion durch 3D-Scanner, dem damit verbundenen Aufwand und der nötigen Expertise zur softwareseitigen Optimierung der Scans, bietet sich dieses Verfahren ohne hochwertige 3D-Scanner nicht an, um Daten für ML-Modelle zu erheben. Ebenso ist das eigenständige Modellieren der 3D-CAD-Modelle nicht nur mit sehr viel Aufwand verbunden, sondern spiegelt möglicherweise nicht die reale Verteilung der Daten wider, auf der schlussendlich die Inferenz stattfindet. Da viele Hersteller der Schaltschrankkomponenten ihre Modelle online frei zur Verfügung stellen bietet das Web-Scraping die beste Lösung zur Aggregation der Daten. Im Rahmen der Arbeit wurde deshalb von der Web Browser Automatisierungsbibliothek Selenium Gebrauch gemacht, um die Daten zu scrapen [80]. Anstelle der Umsetzung entsprechender Browserautomatisierung wäre auch ein Ansatz des direkten Abschickens entsprechender Hypertext Transfer Protocol (HTTP) Anfragen denkbar. Auch wenn die versendeten HTTP-Anfragen einsehbar sind, ist dies für das automatische Scrapen vieler Daten mit sehr viel Aufwand verbunden. Dynamische Websites oder moderne Single Page Applikationen erfordern zumindest teilweise ein Reverse Engineering der API des Backends. 

51 

5 Datenaggregation, Modellierung und Technische Implementierung 

Der Source-Code der Datenaggregation ist dem digitalen Anhang aus dem GitRepository cad-scraper zu entnehmen. Da in der Modellierung von Schaltschränken oft proprietäre Dateiformate wie WSCAD oder EPLAN Anwendung finden, sind diese nicht geeignet aufgrund der fehlenden Möglichkeit die Dateien ohne Lizenz auszulesen. Die für diese Arbeit aggregierten Daten belaufen sich deshalb ausschließlich auf Standard for the Exchange of Product Model Data (STEP) Dateien. Diese sind das direkte Resultat aus der Modellierung in einem CADProgramm. Die direkte Einbettung gewisser Metadaten in eine STEP-Datei ist zwar teilweise möglich, jedoch nur bedingt sinnvoll, aufgrund der ISO 10303-21 Spezifikationen. [81] Da die meisten STEP-Dateien nach dem Download aufgrund mangelnder Informationen nicht mehr den auf der Website enthaltenen Metadaten zugeordnet werden können, wurde die Datenaggregation sequenziell gestaltet, sodass immer ein Modell und anschließend die zugehörigen Metadaten aggregiert wurden. Die Modelle wurden nach einer eindeutigen Produktnummer des Herstellers benannt und in Ordnern für jeden Hersteller separat gespeichert. Dadurch wurde während der Datenaggregation sichergestellt, dass im Falle einer identischen Produktnummer verschiedener Hersteller die Modelle nicht überschrieben werden. Die angesammelten Metadaten wurden in einem assoziativen Array gespeichert, wobei alle dieser Zuordnungstabellen in einem übergeordneten assoziativen Array über die Produktnummer indiziert sind, wodurch der Informationserhalt einer Komponente in konstanter Zeit möglich ist. Zu den aggregierten Metadaten zählen folgende Informationen: 

- Name 

- Hersteller 

- Artikelnummer 

- Bauteiltyp 

- Technologie 

- Kategorie 

- Subkategorie 

- Serie 

- Breite [mm] 

- Höhe [mm] 

- Länge [mm] 

- Strom [A] 

- Artikelstatus 

- Bauteilbeschreibung 

- Externes Dokument 

Die Vollständigkeit der Eigenschaften für alle Komponenten ist nicht gegeben, da nicht alle Hersteller alle genannten Informationen aufführen. Die Eigenschaft „Externes Dokument“ verweist auf die offizielle Herstellerdokumentation der Komponente. Da sich viele Bauteile lediglich in elektrischen Kennwerten 

52 

5 Datenaggregation, Modellierung und Technische Implementierung 

unterscheiden, jedoch eine identische Geometrie aufweisen, wurden zur Identifikation geometrischer Duplikate alle Dateien über einen Secure Hash Algorithm (SHA) 256 Algorithmus gehasht [82]. Ergänzend wurde ein separates Python-Paket entwickelt, das zum Filtern des gesamten Datensatzes auf Basis der Kategorie, Subkategorie und Technologie verwendet werden kann. Das zugehörige Git-Repository ist dem digitalen Anhang D – Digitaler Anhang zu entnehmen und ist als „ccc-dataset“ bezeichnet. 

Da das Aggregieren aller Daten sehr zeitaufwendig ist und ca. 168 Stunden gedauert hat, ist die Nutzung des Pakets auf einem Server, statt Notebook zu empfehlen. Im Rahmen dieser Arbeit wurde hierfür ein Raspberry Pi Model B mit 4 Gigabyte (GB) RAM, in Kombination mit einer externen 2 Terabyte (TB) Hard Disk Drive (HDD) verwendet. Für Server ohne Graphical User Interface kann der Browser über das Python-Paket im Headless-Modus gestartet werden. Zum Aggregieren der Daten wurde der Chrome Browser verwendet, wobei andere Browser durch das Installieren entsprechender Treiber ebenfalls möglich sind. 

## **5.1.2 Datenvorverarbeitung** 

Während der Datenvorverarbeitung wird eine STEP-Datei in ein Dreiecksnetz konvertiert. Dreiecksnetze und Punktwolken lassen sich wesentlich besser als Tensoren darstellen als STEP-Dateien und sind daher besser für DL geeignet. Bisherige Ansätze des DL basierend auf Dreiecksnetzen arbeiten meist ausgehend von Wavefront OBJ (OBJ) oder Object File Format (OFF) Dateien. Das Ziel der Datenvorverarbeitung besteht entsprechend in der effizienten und korrekten Konvertierung und Skalierung der Dateien. Zusätzlich werden Graustufenbilder der Komponenten generiert, welche ebenfalls im Nachgang für 2D-CNNs genutzt werden können. Zuletzt werden die Dreiecksoberflächennetze augmentiert. 

## **Graustufenbilder** 

Die Generierung der Graustufenbilder erfolgt mittels der Open Cascade Technology (OCCT) und rendert sechs Ansichten von oben, unten, vorne, hinten, links und rechts. Aus diesen Ansichten werden dann sechs Bilder mit 512×512 Pixeln erstellt. Da die OCCT-Schnittstelle Farbbilder erstellt, jedoch die Bauteile ohnehin fast ausschließlich in Graufarben modelliert wurden, werden die Bilder noch in Graustufen konvertiert, sodass nur noch ein Farbkanal vorhanden ist. Dadurch wird der Tensor zur Darstellung eines Bildes um den Faktor drei kleiner, indem die Anzahl der Farbkanäle auf einen einzigen minimiert wurde. Dies bietet beispielsweise die Möglichkeit die Batchgröße zu erhöhen, um schneller zu trainieren. Für den praktikablen Einsatz findet das Rendering im Headless-Modus statt, sodass das Ergebnis nicht den Hauptprozess aufhält. Abbildung 28 zeigt beispielhaft die generierten Bilder der STEP-Datei. 

53 

5 Datenaggregation, Modellierung und Technische Implementierung 

**==> picture [339 x 171] intentionally omitted <==**

**----- Start of picture text -----**<br>
oben vorne links<br>unten hinten rechts<br>**----- End of picture text -----**<br>


_Abbildung 28: Graustufenbilder von oben, unten, vorne, hinten, links und rechts der Komponente 1SAE231111M0622_ 

## **Konvertierung** 

Ausgehend von einer STEP-Datei wird zunächst eine Standard Triangle Language (STL) Datei erzeugt. Die STL-Datei dient nur als Zwischenschritt, da alle geläufigen Bibliotheken lediglich die Konvertierung von STEP zu STL nicht jedoch direkt zu Formaten wie OBJ oder OFF unterstützen. Die Konvertierung von STEP zu STL wurde sowohl mittels gmsh als auch über OCCT implementiert. Letzteres Verfahren stellt hierbei sowohl das performantere als auch robustere Konvertierungsverfahren dar. Die folgenden Erläuterungen der Dateiformate stellen lediglich die für diese Arbeit minimalen Informationen dar. Das STL-Format ist beispielhaft in Code 1 dargestellt. Für effizientes Datenmanagement besteht neben dem American Standard Code for Information Interchange (ASCII) Format ebenfalls ein BinärFormat. Wie Code 1 zeigt, wird eine STL-Datei durch Dreiecksfacetten definiert. Es können dabei beliebig viele Dreiecke untereinander definiert werden. Die Angabe der Normalvektoren durch 𝑖 ist dabei optional. Ein Dreieck ist immer aus drei dreidimensionalen Vektoren aufgebaut, welche als „vertex“ oder Scheitelpunkt bezeichnet werden. [83] 

54 

5 Datenaggregation, Modellierung und Technische Implementierung 

_Code 1: Aufbau einer STL-Datei im ASCII-Format nach [83]_ 

```
solid name
  facet normal n1 n2 n3
    outer loop
      vertex p1x p1y p1z
      vertex p2x p2y p2z
      vertex p3x p3y p3z
    endloop
  endfacet
endsolid name
```

Code 2 zeigt den grundlegenden Aufbau einer OBJ-Datei. Auch hier ist die Angabe der Scheitelpunkt- und Dreiecks-Normalvektoren optional. 

## _Code 2: Aufbau einer OBJ-Datei_ 

```
v v1x v1y v1z
v v2x v2y v2z
v v3x v3y v3z
v v4x v4y v4z
vn v1x v1y v1z
vn v2x v2y v2z
vn v3x v3y v3z
vn v4x v4y v4z
f 1//1 2//1 3//1
f 1//2 2//2 4//2
f 1//3 3//3 4//3
f 2//4 3//4 4//4
```

Der beispielhafte Aufbau einer OFF-Datei wird in Code 3 veranschaulicht. Nach dem OFF-Header folgt die Anzahl der Scheitelpunkte, Dreiecke und Kanten des Polygons. Da OFF-Dateien nicht nur Dreiecksnetze, sondern auch Polygone darstellen, werden die Polygone nach den Scheitelpunkten aufgelistet und über die Anzahl der Eckpunkte als Polygon identifiziert. Die Indices der Polygone referenzieren hierbei die Scheitelpunkte. 

## _Code 3: Aufbau einer OFF-Datei_ 

```
OFF
4 4 6
v1x v1y v1z
```

55 

5 Datenaggregation, Modellierung und Technische Implementierung 

```
v2x v2y v2z
v3x v3y v3z
v4x v4y v4z
3 0 1 2
3 0 1 3
3 0 2 3
3 1 2 3
```

Aus den beschriebenen Dateiformaten geht deutlich hervor, dass eine Überführung von OBJ- und OFF-Dateien in eine Tensordarstellung sehr einfach ist, indem ein Tensor alle Scheitelpunkte und ein weiterer Tensor alle Dreiecke beinhaltet. Dies ermöglicht schnelle vektorisierte numerische Berechnungen auf dem gesamten Dreiecksnetz. Im Rahmen der Arbeit erfolgt die Konvertierung vom STL- ins OBJFormat via Open3D. [84] Die Konvertierung von OBJ- zu OFF-Dateien wurde dagegen selbst implementiert, da Open3D hier die Reihenfolge der Scheitelpunkte nicht zwangsläufig beibehält. Zur Gewährleistung korrekter Labels, welche für alle Dateiformate nur ein einziges Mal in identischer Form vorliegen müssen, erfolgt die Konvertierung in OFF-Dateien demnach über eine Implementierung mittels der Python C API. 

## **Skalierung** 

Vor der Konvertierung des OBJ- ins OFF-Format werden die OBJ-Dateien auf eine einheitliche Größe skaliert. Die Skalierung ist aufgrund der stark unterschiedlichen Größe der OBJ-Dateien notwendig. Da das neuronale Netz durch das größte Dreiecksnetz der Trainingsdaten begrenzt ist, gilt es eine geeignete Anzahl an Scheitelpunkten für den gesamten Datensatz zu finden. Mögliche Werte hierfür sind der Median, Mittelwert, das Minimum oder Maximum. Im Falle dieser Arbeit werden alle Dateien auf ca. 6000 Scheitelpunkte skaliert. Dieser Wert ist nicht durch eine der genannten Metriken begründet und wurde stattdessen unter Berücksichtigung des verfügbaren Grafikspeichers gewählt. Da in Kapitel 5.3 mehrere verschiedene Architekturen implementiert werden, ist die Anzahl der Scheitelpunkte durch das ineffizienteste neuronale Netz limitiert. Eine genaue Erläuterung der Anzahl an Scheitelpunkten folgt in Kapitel 5.3.2 unter Berücksichtigung der Funktionsweise des MeshCNN. 

Da während der Konvertierung Fehler auftreten können, gilt es zunächst diese zu korrigieren und erst danach die Skalierung durchzuführen. Das Ziel ist die Generierung einer robusten 2-Mannigfaltigkeit, wobei der Input der Pipeline eine Reihe von Eckpunkten und Dreiecken ist. Die Ausgabe der Pipeline ist eine 2- Mannigfaltigkeit mit ungefähr gleichmäßig auf der Geometriefläche verteilten Scheitelpunkten. Die Umsetzung in dieser Arbeit erfolgt mittels eines Octrees, um das ursprüngliche Netz darzustellen und konstruiert die Oberfläche durch 

56 

5 Datenaggregation, Modellierung und Technische Implementierung 

Isoflächenextraktion. Schließlich erfolgt eine Projektion der Scheitelpunkte auf das ursprüngliche Netz zur Erreichung einer hohen Präzision. [85] Da insbesondere Dreiecksnetze realer Objekte oftmals nicht fehlerfrei sind, ist die Gewährleistung eines wasserdichten Dreiecksnetzes umso wichtiger. Nur unter dieser Voraussetzung ist die Funktionsweise einiger Algorithmen in der Informatik garantiert. Es müssen daher falsche Konnektivitäten, unklare Flächenorientierungen, doppelte Flächen, offene Grenzen und Selbstüberschneidungen im Rahmen der Datenvorverarbeitung behoben werden. Hierfür wird durch die Extraktion äußerer Flächen zwischen belegten und leeren Voxeln und einer projektionsbasierten Optimierungsmethode eine wasserdichte 2-Mannigfaltigkeit wiederhergestellt. [86] Dieses Verfahren ist durch [85] und [86] erarbeitet und auf dem ShapeNet sowie ModelNet10 Datensatz evaluiert worden [58][87]. 

Nach der Sicherstellung dieser Kriterien, kann das Dreiecksnetz vereinfacht werden, um die gewünschte Anzahl von 6000 Scheitelpunkten zu erreichen. Vor dem Anwenden von Downsampling-Algorithmen muss zuerst sichergestellt werden, dass das Dreiecksnetz tatsächlich mehr als die gewünschte Anzahl an Scheitelpunkten aufweist. Anderenfalls wird das darauffolgende Downsampling keine Wirkung zeigen. Eine einfache Netzunterteilung ist hierfür die beste Methode. Dabei wird jedes Dreieck in eine Reihe kleinerer Dreiecke unterteilt. Im trivialsten Fall wird jedes Dreieck in vier Dreiecke unterteilt. Abbildung 29 veranschaulicht, dass bei der Netzunterteilung die Mittelpunkte der drei Kanten die neuen Eckpunkte des inneren Dreiecks bilden. Eine weitere Möglichkeit zur Netzunterteilung ist die Smooth Subdivision. [88] 

**==> picture [454 x 121] intentionally omitted <==**

**----- Start of picture text -----**<br>
Iteration 0 Iteration 1 Iteration 2<br>**----- End of picture text -----**<br>


## _Abbildung 29: Einfache Netzunterteilung nach [84]_ 

Nach genügend Iterationen der Netzunterteilung hat das Dreiecksnetz die gewünschte Anzahl an Scheitelpunkten überschritten, sodass die Vereinfachung des Netzes auf die Zielgröße erfolgen kann. Die Vereinfachung kann über verschiedene Algorithmen erfolgen. Zu diesen zählen unter anderem das Scheitelpunkt-Clusteringund die Netzdezimierung. Ersteres fasst alle in ein Voxel fallende Scheitelpunkte zu einem einzigen Scheitelpunkt zusammen. Der Grad der Vereinfachung hängt hier von der gewählten Voxelgröße ab. Letzteres Verfahren erfolgt in inkrementellen Schritten, indem schrittweise alle Dreiecke entfernt werden, die eine Fehlermetrik 

57 

5 Datenaggregation, Modellierung und Technische Implementierung 

reduzieren, bis die Zielgröße des Dreiecksnetzes erreicht ist. Im Rahmen der Arbeit wird die Vereinfachung nach [85] implementiert. Abbildung 30 zeigt ein reales Beispiel der einfachen und glatten Unterteilung exemplarisch an einer Reihenklemme des Datensatzes. 

**==> picture [451 x 482] intentionally omitted <==**

**----- Start of picture text -----**<br>
Einfache Unterteilung Glatte Unterteilung<br>Eine Iteration<br>wp<br>Zwei Iterationen<br>Ppp<br>Abbildung 30: Vergleich zwischen einfacher und glatter Netzunterteilung am<br>Beispiel einer Reihenklemme<br>**----- End of picture text -----**<br>


Da alle Dreiecksnetze nach der Skalierung gelabelt werden, besteht keine direkte Möglichkeit mittels dieser Labels neuronale Netze mit einer höheren Anzahl an Scheitelpunkten zu trainieren. Hierfür müssen sowohl die Dreiecksnetze selbst als auch die Labels hochskaliert werden. Das Hochskalieren kann dabei normal über die beschriebenen Verfahren der Netzunterteilung erfolgen. Für die einfache Netzunterteilung wurde das Hochskalieren der fehlenden Labels neuer Dreiecke implementiert, sodass diese künstlich generiert werden. Die einfachste Methode ist 

58 

5 Datenaggregation, Modellierung und Technische Implementierung 

allen neuen Dreiecken aus Iteration die Labels aus Iteration −1 zuzuordnen, welche dem umschließenden Dreieck angehören. Für Dreiecksnetze des gleichen Bauteils, welche sich grundlegend in der Struktur aufgrund eines anderen MeshAlgorithmus unterscheiden, wurde ein Algorithmus zur Approximation der Labels implementiert. Hierfür wird jedem Scheitelpunkt des neuen Dreiecksnetzes dasjenige Label des Ursprungsnetzes zugeordnet dessen Frobeniusnorm minimal ist. 

## **Augmentierung** 

Datenaugmentierung ist ein grundlegendes Konzept in der Datenvorverarbeitung für das Trainieren von ML-Modellen. Für Dreiecksnetze wurden folgende drei unterschiedliche und kombinierbare Verfahren zur Datenaugmentierung implementiert: 

- Zufällige Rotationen 

- Rauschen 

- Dreiecksdeformationen 

Zufällige Rotationen wurden durch das Anwenden einer Rotationsmatrix erzeugt. Die Rotationen richten sich nach den Winkeln, welche aus einer stetigen Gleichverteilung entnommen werden. Das Rauschen variiert die Vektoren um einen kleinen Grad in zufällige Richtungen. Die Werte zur Manipulation der Vektoren basieren auf einer logarithmischen Normalverteilung, da ausschließlich positive Werte benötigt werden, um die Vorzeichen des Vektors beizubehalten. Als Mittelwert µ und Standardabweichung _σ_ wurden jeweils 0 und 0,005 gewählt. Daraus resultiert die in Abbildung 31 dargestellte Wahrscheinlichkeitsverteilung. Durch das Generieren eines Tensors der identischen Größe des Scheitelpunkt-Tensors und der anschließenden elementweisen Multiplikation beider Tensoren ergibt sich ein Dreiecksnetz mit Rauschen basierend auf der logarithmischen Normalverteilung. Nach dem Hinzufügen des Rauschens wird das Dreiecksnetz geglättet. Hierfür kann ein Durchschnitts-, Laplace- oder Taubin-Filter verwendet werden. [84][89] Beim Durchschnittsfilter aus Formel 5.1 wird ein Scheitelpunkt durch den Durchschnitt der benachbarten Scheitelpunkte 𝑁 berechnet. 

**==> picture [300 x 29] intentionally omitted <==**

Der Laplace-Filter ist durch Formel 5.2 definiert. 

**==> picture [317 x 31] intentionally omitted <==**

Hierbei gibt 𝜆 die Stärke des Filters an und 𝑤𝑛 repräsentiert normalisierte Gewichte, die sich auf den Abstand der benachbarten Scheitelpunkte beziehen. 

59 

5 Datenaggregation, Modellierung und Technische Implementierung 

**==> picture [454 x 283] intentionally omitted <==**

**----- Start of picture text -----**<br>
90<br>80<br>70<br>60<br>50<br>40<br>30<br>20<br>10<br>0<br>0.980 0.985 0.990 0.995 1.000 1.005 1.010 1.015 1.020<br>x →<br>**----- End of picture text -----**<br>


_Abbildung 31: Dichtefunktion der logarithmischen Normalfunktion mit µ = 0 und σ = 0,005_ 

Die letzte Möglichkeit zur Augmentierung der Dreiecksnetze liegt in der bewussten Deformation dieser. Hierfür wurde der As-Rigid-As-Possible-Algorithmus verwendet. [90] Im Rahmen dieser Arbeit werden zufällig zwischen eins und zehn Punkten in den vier geometrischen Featuregruppen Kontaktierung bzw. Werkzeugeinschub, Aufrastpunkt, Kabeleinführung und Beschriftungsfläche ausgewählt. Eine Verformung des Gehäuses wird bewusst vermieden, da dieses nur als Hintergrundklasse dient und kleine Verformungen des Gehäuses aus den Deformationen der Features ohnehin resultieren. Von den ausgewählten Punkten wird anschließend die Frobeniusnorm zu allen Scheitelpunkten des Netzes sowie das 50 %-Perzentil der Norm berechnet. Alle Vektoren deren Abstand über dem 50 %-Perzentil liegen werden den statischen Scheitelpunkten zugeordnet. Dann erfolgt die Deformation des Punktes mittels des As-Rigid-As-Possible Algorithmus. [90] Diese Schritte werden mehrmals wiederholt, bis die gewünschte Anzahl an Deformationen erreicht ist. Abbildung 32 stellt Teile der Datenaugmentierung dar, wobei (a) die normale Reihenklemme darstellt, bei (b) Rauschen hinzugefügt wurde und in (c) die durch das Rauschen enstandenen Zacken über einen Laplace-Filter geglättet und anschließend mehrere As-Rigid-As-Possible-Deformationen durchgeführt werden. 

60 

5 Datenaggregation, Modellierung und Technische Implementierung 

**==> picture [13 x 426] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a)<br>(b)<br>(c)<br>**----- End of picture text -----**<br>


_Abbildung 32: Vergleich zwischen normaler Reihenklemme (a), mit Rauschen (b) und mit Laplace-Filter und As-Rigid-As-Possible-Deformationen (c)_ 

61 

5 Datenaggregation, Modellierung und Technische Implementierung 

In Vergleich zu (a) sind in (c) die Deformationen der Kabeleinführung und der Beschriftungsfläche klar zu erkennen. Die Parameter zur Einstellung der Stärke der Deformationen wurden so gewählt, dass die topologischen Eigenschaften der 2- Mannigfaltigkeiten erhalten bleiben und sich dadurch die Invarianten nicht ändern. 

## **5.2 Labeling der Daten** 

Da die in dieser Arbeit verfolgten Ansätze in den Bereich des überwachten Maschinellen Lernens fallen, erfolgt nach der Konvertierung und Skalierung der Dreiecksnetze das Labeling dieser. Als Grundlage für das Anwenden von 2D-CNNs werden zusätzlich die in Kapitel 5.1.2 erzeugten Graustufenbilder gelabelt. 

## **5.2.1 Dreiecksnetzlabels** 

Während für verschiedene Datentypen meist eine Vielzahl verschiedener LabelingTools zur Verfügung steht, ist dies bei Dreiecksnetzen nicht der Fall. Hier kommen entweder eine selbst implementierte Lösung basierend auf Open3D in Frage oder die Verwendung von Grafik-Suites wie Blender. Letzteres ist eine kostenfreie opensource Software zur Bearbeitung und Animation von 3D-Objekten. [91][92] Vor dem Labeln wird zuerst die Struktur der Labels festgelegt. Bei der Verarbeitung von Dreiecksnetzen mittels ML bestehen drei verschiedene Ansätze zum Labeln der Daten: 

- Labeln der Knoten 

- Labeln der Kanten 

- Labeln der Dreiecksflächen 

Da aus dem Labeln nach einem dieser Prinzipien beide weiteren Labels durch einfache Konvertierung abgeleitet werden können, spielt die Vorgehensweise eine untergeordnete Rolle. Im Rahmen dieser Arbeit wurde als Labeling-Tool Blender verwendet. Das Labeling erfolgte in Blender über das Zuordnen der Knoten zu den aufgelisteten Segmentierungsklassen: 

- Gehäuse 

- Kontaktierung 

- Aufrastpunkt 

- Kabeleinführung 

- Beschriftungsfläche 

Die Segmentierung kann in Blender via Vertex-Gruppen umgesetzt werden. Da diese Arbeit unter anderem Ansätze verfolgt, welche jeweils genau ein valides Label pro Vertex zulassen, entspricht die Summe der Scheitelpunkte aller Vertex-Gruppen der Gesamtanzahl der Scheitelpunkte des Dreiecksnetz. Dieser Ansatz wurde durch [61] 

62 

5 Datenaggregation, Modellierung und Technische Implementierung 

vorgeschlagen und ebenfalls erfolgreich in [60] umgesetzt. Abbildung 33 zeigt eine gelabelte Komponente in Blender. 

## _Abbildung 33: Gelabelte Vertices einer Reihenklemme in Blender_ 

Gegensätzlich zu diesen Ansätzen, welche das Ausführen von Python-Skripten im Python-Terminal innerhalb von Blender erfordern, wurde im Rahmen dieser Arbeit eine Schnittstelle zur automatisierten Extraktion der Labels programmiert. Dies ermöglicht zum einen die Separierung des Labelings und der Datenvorverarbeitung und zum anderen den automatischen Export, ohne Blender selbst zu öffnen, manuell Skripte zu kopieren und anschließend auszuführen. Zur Umsetzung der Schnittstelle wurde das Python-Paket bpy verwendet. Da Blender in C geschrieben ist und lediglich Python-Bindings aufweist muss das gesamte Projekt zuerst kompiliert werden. Hierfür wurde Blender Version 3.3.0 verwendet. Die kompilierte SharedLibrary wurde anschließend in einer virtuellen Conda-Umgebung integriert. Die hierfür verwendete Version des Python-Interpreters ist 3.10.2. Durch diese Schnittstelle können die Labels der Knoten innerhalb kurzer Zeit vollständig generiert werden, sodass diese als eindimensionaler Tensor vorliegen. Abbildung 34 stellt die 1:1-Zuordnung der Labels zum zweidimensionalen Vertex-Tensor exemplarisch dar. Aufgrund von fünf Segmentierungsklassen nehmen die Labels die Werte { 0, 1, 2, 3, 4 } an. 

63 

5 Datenaggregation, Modellierung und Technische Implementierung 

**==> picture [454 x 91] intentionally omitted <==**

_Abbildung 34: Exemplarische Zuordnung der Scheitelpunkt-Labels zu den Scheitelpunkten_ 

Die Umwandlung der Knoten-Labels in Kanten-Labels wurde identisch zu [60] umgesetzt. Die Konvertierung in Flächenlabels wurde so umgesetzt, dass ein Dreieck als Gehäuse (0) gelabelt wird, wenn nicht alle drei Knoten des Dreiecks identische Labels aufweisen. Anderenfalls wird das für alle drei Knoten identische Label verwendet. Abbildung 35 veranschaulicht den Unterschied der Knoten-, Kanten- und Flächenlabels. 

**==> picture [454 x 146] intentionally omitted <==**

**----- Start of picture text -----**<br>
Knotenlabels Kantenlabels Flächenlabels<br>**----- End of picture text -----**<br>


_Abbildung 35: Vergleich zwischen Knoten-, Kanten-, und Flächenlabels_ 

Nach der Konvertierung der Knotenin Kantenlabels, können die Schaltschrankkomponenten als Dreiecksgitter dargestellt werden. 

## **5.2.2 Labels der Graustufenbilder** 

Die während der Datenvorverarbeitungsphase generierten Graustufenbilder werden mit Bounding-Boxen gelabelt. Abbildung 36 stellt exemplarisch die Labels der Ansichten von oben, vorne und links einer Schaltschrankkomponente dar. Dieses Beispiel veranschaulicht ebenfalls, dass nicht alle Komponenten über alle Labelkategorien verfügen. Die hier dargestellte Komponente weist beispielsweise keine Beschriftungsflächen auf. 

64 

5 Datenaggregation, Modellierung und Technische Implementierung 

**==> picture [412 x 165] intentionally omitted <==**

**----- Start of picture text -----**<br>
= Kontaktierung o Aufrastpunkt Kabeleinführung m<br>**----- End of picture text -----**<br>


_Abbildung 36: Exemplarische Labels der Features mittels Bounding-Boxen der Komponte 1SAE231111M0622_ 

Das Labeln der Bilder erfolgte mittels der open-source Software Label Studio. Abgesehen von den Bounding-Boxen wurde zudem die Ansicht gelabelt. [93] Die Labels von allen Bildern wurden danach in jeweils separate Dateien exportiert um für jede Bounding-Box die x- und y-Koordinate, sowie die Höhe und Breite festzuhalten. Code 4 stellt ein pseudohaftes Beispiel der Labels dar. Aus diesem Format wird das Common Objects in Context (COCO) Format für Objekterkennung als JavaScript Object Notation (JSON) Datei generiert. [94] 

_Code 4: Pseudolabels eines Bauteils mit einer Kontaktierung (1), einem Aufrastpunkt (3), einer Kabeleinführung (0), sowie einer Beschriftungsfläche (2)_ `0 0.385742187 0.639648437 0.041207188 0.045113438 1 0.386199209 0.221591701 0.053012202 0.052183886 2 0.544821658 0.768485772 0.050527255 0.0505272553 3 0.340820312 0.728515625 0.025390625 0.023437541` ~~OB~~ **5.3 Modellierung künstlich neuronaler Netze** 

Zur Überwindung der nicht-intrinsischen Datenoperationen implementiert dieses Kapitel neuronale Netze zur Segmentierung von Oberflächendreiecksnetzen. Es wird zunächst auf eine Architektur und Abwandlung dieser zur Kantensegmentierung nach [61] und [60] eingegangen. Anschließend folgt eine oberflächenunabhängige Implementierung zum Überwinden der Probleme vorheriger Architekturen. [56] Danach werden zwei verschiedene Verfahren zur Aufteilung der Segmentierungsmasken in einzelne Instanzen umgesetzt. 

65 

5 Datenaggregation, Modellierung und Technische Implementierung 

## **5.3.1 MeshCNN** 

Aufgrund der erfolgreichen Anwendung von CNNs auf Klassifikations- und Segmentierungsaufgaben liegt der Einsatz dieser nahe. Da im Gegensatz zu Bildern Dreiecksnetze keine regelmäßige Struktur aufweisen ist die Erweiterung von CNNs auf Dreiecksnetze jedoch nicht trivial. Die Projektion herkömmlicher regelmäßiger Faltungen auf das unregelmäßige Dreiecksnetz ist speicherineffizient und führt redundante Berechnungen durch. Es wird deshalb eine spezifische Implementierung der Faltungs- und Pooling-Operation benötigt, um beispielsweise die Faltung leerer Voxel zu vermeiden. Ein MeshCNN nach [61] arbeitet direkt auf unregelmäßigen Dreiecksnetzen und führt die Faltungs- und Pooling-Operationen intrinsisch aus. Die Kanten eines Netzes können dabei als äquivalent zu Pixeln in Bildern angesehen werden. In konventionellen Techniken zur Netzvereinfachung werden diejenigen Kanten eliminiert, welche die geometrische Verzerrung minimieren. Beim MeshPooling werden dagegen Kanten bereinigt, deren Merkmale am wenigsten zum angestrebten Ziel beitragen. Zur Erhöhung der Flexiblität und Unterstützung einer Vielzahl von Daten vereinfacht die Pooling-Schicht das Netz auf eine zuvor festgelegte konstante Anzahl an Kanten. 

## **Invariante Faltungen** 

Aus der Annahme, dass bei 2-mannigfaltigen Dreiecksnetzen jede Kante Bestandteil maximal zweier Flächen ist, ergeben sich daher zwei oder vier weitere angrenzende Kanten. Bei einer Anordnung der Scheitelpunkte gegen den Uhrzeigersinn ergeben sich zwei mögliche Anordnungen der vier Nachbarn jeder Kante. Abbildung 37 veranschaulicht dies, indem die Nachbarn von e als (abcd) oder (cdab) abhängig von der Wahl des ersten Nachbarn angeordnet werden können. Da dies zu einer Mehrdeutigkeit des rezeptiven Felds der Faltung führt und die Bildung von invarianten Merkmalen behindert, werden zwei Maßnahmen ergriffen, um die Invarianz gegenüber Ähnlichkeitstransformationen wie Rotation, Translation und Skalierung zu gewährleisten. Erstens wir der Eingangsdeskriptor einer Kante so gewählt, dass dieser nur relative Merkmale enthält, die automatisch invariant gegenüber Ähnlichkeitstransformationen sind. Zweitens werden die vier Ringkanten aller mehrdeutiger Kanten jeweils zu zwei Kantenpaaren zusammengefasst, z.B. a und c sowie b und d. Das Erzeugen neuer Merkmale erfolgt dann durch symmetrische Funktionen auf jedem dieser Paare, z.B. die Summe aus a und c. Die Faltung wird auf die neuen symmetrischen Merkmale angewandt, wodurch sämtliche Mehrdeutigkeiten der Faltung beseitigt sind. [61] 

66 

5 Datenaggregation, Modellierung und Technische Implementierung 

**==> picture [454 x 143] intentionally omitted <==**

**----- Start of picture text -----**<br>
𝑏<br>𝑎<br>𝑒<br>𝑐<br>𝑑<br>Mesh Convolution<br>**----- End of picture text -----**<br>


## _Abbildung 37: Invariante Faltung auf Dreiecksnetzen nach [61]_ 

Aus Abbildung 37 ergibt sich für die Faltung mit einem Kernel 𝑘 und der Nachbarschaft der Kanten für das Kantenmerkmal 𝑒 Formel 5.3: [61] 

**==> picture [300 x 41] intentionally omitted <==**

Hierbei ist 𝑒[𝑗] das Merkmal der 𝑗 -ten Nachbarn von 𝑒 [61]. 

## **Eingangsmerkmale** 

Das in Abbildung 38 dargestellte Eingangsmerkmal ist jeweils als fünfdimensionaler Vektor je Kante definiert und besteht aus dem Flächenwinkel, zwei Innenwinkeln und zwei Kantenlängenverhältnissen für jede Fläche. Das Kantenverhältnis ist hierbei das Verhältnis zwischen der Länge der Kante und der senkrechten Linie für jede benachbarte Fläche. Durch das Sortieren jedes der beiden flächenbasierten Merkmale (Innenwinkel und Kantenverhältnis) wird die Ordnungszweideutigkeit gelöst und Invarianz garantiert. Durch die Relativität der Merkmale ändern sich diese nicht in Bezug auf Translationen, Rotationen und Skalierungen. [61] 

**==> picture [454 x 128] intentionally omitted <==**

## _Abbildung 38: Eingangsmerkmale des MeshCNNs nach [61]_ 

67 

5 Datenaggregation, Modellierung und Technische Implementierung 

## **Mesh Pooling** 

Der im MeshCNN durchgeführte Kantenkollaps wird als Mesh Pooling bezeichnet. Wie in Abbildung 39 dargestellt, kollabiert die orange Kante zu einem Punkt und anschließend verschmelzen jeweils beide blauen sowie grünen Kanten. Beim Kollaps einer Kante werden demnach drei Kanten eliminiert. Die Priorisierung der Kanten, welche durch die Mesh Pooling Operation eliminiert werden, erfolgt dabei durch die Länge der Kanten, sodass informativere Kanten länger erhalten bleiben. Kürzere Kanten werden deshalb früher eliminiert. [61] 

**==> picture [454 x 155] intentionally omitted <==**

**----- Start of picture text -----**<br>
𝑝 =  𝑎𝑣𝑔(𝑎,𝑏, 𝑒)<br>𝑏<br>𝑝<br>Mesh Pooling Mesh Unpooling<br>𝑎 𝑝<br>𝑒<br>𝑐<br>𝑎𝑣𝑔(𝑝,  )<br>𝑑<br>  =  𝑎𝑣𝑔(𝑐, 𝑑,𝑒)<br>**----- End of picture text -----**<br>


## _Abbildung 39: Mesh Pooling und Mesh Unpooling nach [61]_ 

Aufgrund der Aufteilung des MeshCNNs in einen Encoder und Decoder ist zudem eine Unpooling-Operation nötig, welche die Pooling-Operation teilweise umkehrt. Während Pooling-Schichten durch Informationskodierung und -komprimierung die Merkmalsaktivierungen reduzieren, erhöhen die Unpooling-Schichten die Auflösung der Merkmalsaktivierungen durch Informationsdekodierung und -dekomprimierung. Im Encoder wird hierfür eine Historie der Pooling-Operationen und betroffenen Kanten gecacht. Unter Zuhilfenahme des Cache kann der Decoder die UnpoolingOperation durchführen. Folglich erfordert das Unpooling keine lernbaren Parameter und kann deshalb mit Faltungen zur Wiederherstellung der ursprünglichen Auflösung kombiniert werden. Jede Unpooling-Schicht im Decoder ist mit einer Pooling-Schicht im Encoder gepaart, um die Topologie und Kantenmerkmale des Dreiecksnetz zu verbessern. [61] 

## **5.3.2 MedMeshCNN** 

Als Abwandlung des MeshCNNs wurden ebenfalls Bestandteile des nach medizinischen Anwendungen benannten MedMeshCNN implementiert. Dieses unterscheidet sich vom vorherigen Kapitel lediglich in der softwareseitigen Implementierung, jedoch nicht in der logischen Umsetzung. Während das MeshCNN zur Darstellung der Kanten eine Adjazenzmatrix verwendet, wird diese im 

68 

5 Datenaggregation, Modellierung und Technische Implementierung 

MedMeshCNN bewusst vermieden. Da die Adjazenzmatrix fast ausschließlich Nullen enthält, wird diese stattdessen durch eine spärliche Matrix ersetzt. Ein Beispiel einer spärlichen Matrix ist durch Formel 5.4 gegeben. Hierbei stellt die erste Spalte von 𝑀𝑠𝑝𝑎𝑟𝑠𝑒 den Zeilenindex und die zweite Spalte den Spaltenindex von 𝑀𝑑𝑒𝑛𝑠𝑒 dar. Die dritte Spalte enthält dann den entsprechenden Wert. [33][95]. 

**==> picture [374 x 52] intentionally omitted <==**

Durch den geringeren benötigten Speicher können entsprechend größere Dreiecksnetze verarbeitet werden. Die im Rahmen dieser Arbeit gewählte Anzahl an Scheitelpunkten je Dreiecksnetz ergibt sich durch die Restriktionen der Adjazenzmatrix des MeshCNNs. Bei 6000 Scheitelpunkten besteht ein Dreiecksnetz nach der Euler-Charakteristik entsprechend aus 18000 Kanten, wodurch eine Adjazenzmatrix der Größe (18000, 18000) nötig ist. [53] Der dadurch aggregierte Speicher des Tensors auf der in dieser Arbeit verwendeten GPU, beansprucht demnach wie in Code 5 berechnet einen Großteil des gesamten GPU-Speichers. [33] 

_Code 5: Berechnung des aggregierten CUDA-Speichers in Bytes in Python mittels PyTorch nach [33]_ 

```
> import torch
```

```
> t = torch.ones((18000, 18000), dtype=torch.int)
```

```
> t.element_size() * t.nelement()
1296000000
```

Der benötigte Grafikspeicher beläuft sich daher auf ca. 1,3 GB. Da etliche weitere Tensoren für das Mesh Pooling und Mesh Unpooling benötigt werden leidet darunter die Batchgröße. Bei 6000 Scheitelpunkten und drei ResNet-Blöcken ist deshalb gerade noch eine Batchgröße von zwei möglich. Obwohl MedMeshCNN aufgrund der Verwendung einer spärlichen Matrix speichereffizienter als das MeshCNN ist, werden alle im Rahmen dieser Arbeit erprobten Modelle aus Gründen der Vergleichbarkeit auf Dreiecksnetzen mit 6000 Scheitelpunkten trainiert. 

## **5.3.3 DiffusionNet** 

Sowohl das MeshCNN als auch dessen Abwandlung MedMeshCNN sind aufgrund der Mesh-Pooling-Operation sehr speicher- und rechenintensiv. Dieses Kapitel setzt einen alternativen Ansatz basierend auf einer Diffusionsschicht um, welcher beide Probleme löst, um auch größere Dreiecksnetze verwenden zu können und wird als DiffusionNet bezeichnet. Das Kernproblem vieler Methoden des oberflächenbasierten Lernens besteht in einer ungeeigneten technischen 

69 

5 Datenaggregation, Modellierung und Technische Implementierung 

Implementierung der Faltungs- und Pooling-Operation. Im Falle nicht-euklidischer Daten ist eine universelle Oberflächenfaltung bisher nicht möglich. Entgegen der Faltung von Kanten wird nun ein ANN basierend auf der Berechnung von Geodäten und parallelem Transport implementiert. Der Kerngedanke besteht im Ersetzen zuvor rechenintensiver Operationen durch zwei geometrische Mechanismen. Diese sind eine gelernte Diffusionsschicht zur Informationsausbreitung und ein räumlicher Gradient zur Erfassung der Anisotropie. Über das Diskretisieren mittels Techniken der Differentialgeometrie wird dadurch ein skalierbareres und robusteres Netzwerk möglich. Das DiffusionNet besteht aus den drei Grundbausteinen MLP, Diffusionsschicht und Gradientenmerkmale. MLPs werden zur punktweisen skalaren Modellierung an jedem Punkt angewendet. Die erlernte Diffusionsschicht wird zur Ausbreitung von Informationen über die Domäne verwendet. Die räumlichen Gradientenmerkmale erweitern den Filterraum des Netzwerks über radialsymmetrische Filter. [56] 

## **Architektur** 

Die Kombination dieser drei Bestandteile verkörpert einen Diffusionsblock, wobei die Aneinanderreihung mehrerer Diffusionsblöcke das DiffusionNet bildet. Dieses Netzwerk arbeitet durchgängig mit einer fixen Kanalbreite von Skalarwerten, wobei jeder Diffusionsblock die Merkmale diffundiert, räumliche Gradientenmerkmale konstruiert und das Ergebnis an ein MLP weiterleitet. Es werden zudem SkipVerbindungen zur Stabilisierung des Trainings sowie lineare Schichten zur Umwandlung in die erwartete Eingangs- und Ausgangsdimension umgesetzt. Gegebenenfalls können die Ergebnisse an den Kanten oder Flächen eines Netzes durch Mittelung des Outputs von den einfallenden Scheitelpunkten berechnet werden, um beispielsweise Flächen statt Scheitelpunkten zu segmentieren. Da diese Arbeit das Ziel der Segmentierung von Dreiecksnetzen verfolgt, wird eine SoftmaxAktivierungsfunktion am Ende angehängt. Abbildung 40 stellt die grundlegende Architektur des DiffusionNets dar. Als Inputmerkmale nimmt das DiffusionNet einen Vektor skalarer Werte pro Scheitelpunkt an. Der Vektor der Merkmale entspricht dem in Kapitel 5.2.1 erläuterten Format. Der Input der Scheitelpunkte selbst kann entweder über die Darstellung als rohe 3D-Koordinaten oder aber als Heat Kernel Signatur (HKS) erfolgen. Da die Daten nicht konsistent ausgerichtet sind, werden zur Förderung der Invarianz Rotationserweiterungen verwendet. Im Falle nicht-starrer Invarianz empfiehlt sich stattdessen die HKS, welche jedoch in dieser Arbeit von geringerer Bedeutung ist. [56] 

70 

5 Datenaggregation, Modellierung und Technische Implementierung 

**==> picture [688 x 418] intentionally omitted <==**

**----- Start of picture text -----**<br>
Diffusionsblock Berechnung der Diffusion<br>𝐴𝑑𝑑𝑖𝑡𝑖𝑜<br>impliziter Zeitschritt<br>Skalarwerte je<br>Scheitelpunkt<br>ℎ𝑡 𝑢 (𝑀+ 𝑡 ) [−1] 𝑀𝑢<br>oder<br> 𝑒𝑟𝑘𝑒𝑡𝑡𝑢 𝑔<br>schnelle Spektralanalyse<br>räumliche<br>Diffu sion<br>Räumliche Merkmale<br>ℎ𝑡(𝑢) des Gradienten 𝑀   𝑗𝑒  𝑐ℎ𝑒𝑖𝑡𝑒𝑙𝑝𝑢 𝑘𝑡<br>[3N,N,N,N]<br> 𝑟𝑙𝑒 𝑡𝑒𝑠 𝑡   [  ] 𝑒𝑟𝑙𝑒𝑟 𝑡𝑒𝑟  𝑒𝑤𝑖𝑐ℎ𝑡𝑒<br>𝑗𝑒  𝑎 𝑎𝑙<br>𝑡𝑎 ℎ( 𝑒(𝑤  𝐴𝑤))<br> 𝑟𝑙𝑒 𝑡𝑒𝑠 𝐴 𝐵𝑟𝑒𝑖𝑡𝑒: 𝑁 Cache<br>Cache Berechnung<br>Laplace, Massenmatrix  ,𝑀  ,𝑀<br> 𝜙𝑖<br>Räumliche Gradientenmatrix<br> 𝜙𝑖 = 𝜆𝑖𝑀𝜙𝑖<br>𝜆, 𝜙<br>Eigenbasis 𝜆, 𝜙  𝑖𝑔𝑒 𝑏𝑎𝑠𝑖𝑠<br>DiffusionNet<br>Input Output<br>Diffusionsblock Diffusionsblock . . . Diffusionsblock<br>𝑢 𝐴𝑖𝑛𝑢 𝑢 𝐴𝑜𝑢𝑡𝑢<br>**----- End of picture text -----**<br>


_Abbildung 40: DiffusionNet Architektur nach [56]_ 

71 

5 Datenaggregation, Modellierung und Technische Implementierung 

## **5.3.4 YOLOv6** 

Zur Identifikation der Anzahl der Instanzen je Merkmal wurde ein YOLO-Netzwerk implementiert. Die YOLO-Serie ist aufgrund der Balance zwischen Geschwindigkeit und Genauigkeit das etablierteste Detektionssystem in industriellen Anwendungen. Während das YOLO-Framework allgemein in den ersten drei Versionen einen neuartigen Ansatz für einstufige Detektoren beschrieben hat, wurde das Framework in Version vier in separate Bestandteile aufgegliedert, um verschiedene NetzwerkBackbones zu integrieren. [96] Die zum Zeitpunkt der Arbeit erfolgreichste Version der YOLO-Serie ist YOLOv6, welches im Rahmen dieser Arbeit implementiert wurde. Ein einstufiger Objektdetektor besteht im Allgemeinen aus den drei Teilen, Rückgrat, Hals und Kopf. Das Rückgrat bestimmt hauptsächlich die Fähigkeit der Merkmalsdarstellung. Aufgrund des hohen Anteils des Gerüsts an der gesamten Rechenlast hat dieses einen wesentlichen Einfluss auf die Inferenzzeit. [97] Der zweite Teil dient zur Kombination physischer Merkmale niedriger Ebenen mit den semantischen Merkmalen auf höheren Ebenen. Der dritte Teil beinhaltet mehrere Faltungsschichten und dient zur Prognose der endgültigen Ergebnisse anhand von Merkmalen der zuvor durchlaufenen Ebenen. Die gesamte Architektur des verwendeten YOLO-Netzwerks ist in Abbildung 41 dargestellt. [98] YOLOv6 baut auf der Reparametrization (Rep) Visual Geometry Group (VGG) auf, weshalb Abbildung 42 die Rep-Blöcke aus Abbildung 41 detailliert illustriert [99]. Da die Objekterkennung grundsätzlich sowohl die Klassifizierung als auch die Lokalisierung umfasst, werden zwei Verlustfunktionen benötigt. Der effizient entkoppelte Kopf am Ende des Netzwerks aus YOLOv5 trennt das Netzwerk in Klassifikationsund Regressionsverlustfunktionen auf. [100] Zur Klassifikation wird die VariFocalVerlustfunktion verwendet und zur Regression eine Variante der Intersection over Union (IoU) Verlustfunktion. [98] Die Wahl dieser Netzwerkarchitektur beruht insbesondere auf der zugrundeliegenden RepVGG-Architektur, welche einfach gehalten, schnell und speichereffizient ist [99]. 

72 

5 Datenaggregation, Modellierung und Technische Implementierung 

**==> picture [680 x 367] intentionally omitted <==**

**----- Start of picture text -----**<br>
Rep-PAN<br>Regression<br>Verkettung über<br>U Upsampling C<br>Dimension der Kanäle<br>Effizient-<br>C entkoppelter<br>Kopf<br>Klassifikation<br>Input<br>U Conv Conv<br>Regression<br>Effizient-<br>EfficientRep<br>C RepBlock C RepBlock entkoppelter<br>Rückgrat<br>Kopf<br>le<br>Klassifikation<br>U Conv Conv<br>Regression<br>Effizient-<br>C RepBlock entkoppelter<br>Kopf<br>Klassifikation<br>**----- End of picture text -----**<br>


_Abbildung 41: YOLOv6 Architektur nach [98]_ 

73 

5 Datenaggregation, Modellierung und Technische Implementierung 

**==> picture [454 x 399] intentionally omitted <==**

**----- Start of picture text -----**<br>
3 x 3 1 x 1 3 x 3<br>+<br>3 x 3 1 x 1 3 x 3<br>+<br>3 x 3 1 x 1 3 x 3<br>+<br>3 x 3 1 x 1 3 x 3<br>+<br>Faltung<br>RepVGG RepVGG<br>Training Inferenz<br>ReLU<br>**----- End of picture text -----**<br>


_Abbildung 42: RepVGG Architektur nach [99]_ 

## **5.3.5 Instanziierung der Segmentierungsmasken** 

Nach der Segmentierung der Cluster in einzelne Klassen liegt der nächste Schritt in der Identifikation einzelner Instanzen. Hierfür wurden zwei verschiedene Methoden implementiert. Zum einen können Segmentierungsmasken mittels normalen ClusterAlgorithmen untergliedert werden und zum anderen über die Zuhilfenahme der Graphkonnektivität der Dreiecksnetze. 

## **Instanziierung mittels Clusteranalyse** 

Im Rahmen dieser Arbeit wurde der k-Means-Algorithmus verwendet. Da dieser die Anzahl der Cluster sowohl selbst ermitteln kann als auch unter Vorgabe der Zielcluster funktioniert, wurden beide Ansätze ausprobiert. Die Anzahl der Zielcluster wurde demnach durch das im vorherigen Kapitel beschriebene YOLO-Netzwerk 

74 

5 Datenaggregation, Modellierung und Technische Implementierung 

ermittelt. Das Ziel des k-Means-Algorithmus liegt in der Optimierung der quadratischen Abweichung um den Mittelwert. [101] 

## **Instanziierung über Graphkonnektivität** 

Als Alternative kann nach der Zuweisung der segmentierten Labels zu den Scheitelpunkten das Dreiecksnetz in alle Klassen untergliedert werden. Die Konnektivität des resultierenden Graphen muss dabei nicht zwangsläufig erhalten bleiben. Alle Teilgraphen, welche nicht miteinander verbunden sind, werden hierbei als separate Instanz identifiziert. Insbesondere beim Anwenden dieses Verfahrens auf den Vorhersagen eines der vorgestellten neuronalen Netze können bei falschen Segmentierungen sehr kleine und viele Cluster entstehen. Da korrekt identifizierte Merkmale üblicherweise eine Mindestanzahl an Scheitelpunkten erfordern, werden Cluster unter einer einem gewissen Schwellwert verworfen und standardmäßig dem Gehäuse zugeordnet. Abbildung 43 veranschaulicht das Prinzip der Graphkonnektivität beispielhaft. Da zusätzlich Probleme in der Separierung sehr nah aneinander liegender Merkmale entstehen können, wurde die äußerste Schicht der Graphen der einzelnen Merkmale entfernt. Die Eliminierung der Labels erfolgt dabei auf Basis der Nachbarlabels. Alle Scheitelpunkte, welche nicht ausschließlich von identischen Labels umgeben sind, wurden dem Gehäuse zugeordnet. 

**==> picture [444 x 224] intentionally omitted <==**

**----- Start of picture text -----**<br>
B B<br>A A<br>D D<br>C C<br>Klasse 0<br>(a) Klasse 1 (b)<br>Grenzscheitelpunkte<br>**----- End of picture text -----**<br>


_Abbildung 43: Zuweisen von Scheitelpunkten an den Grenzen der Merkmale zur Standardklasse_ 

Aus Abbildung 43 wird deutlich, dass die zwei in (a) verbundenen Cluster A und B der Klasse 1 nach dem Zuweisen der Grenzscheitelpunkte zur Standardklasse 0 in (b) nicht mehr miteinander verbunden sind. Sehr kleine Cluster der Klasse 1 in (a) wie beispielsweise C werden ohnehin eliminiert. Auf alle in (b) weiterhin bestehenden 

75 

5 Datenaggregation, Modellierung und Technische Implementierung 

Cluster A, B und D kann anschließend ein Filter angewendet werden, um nur noch Cluster zu betrachten, deren Anzahl an Scheitelpunkten über dem spezifizierten Schwellwert liegen. Bei einem Schwellwert von drei, würden insgesamt die zwei Instanzen A und B der Klasse 1 identifiziert werden. Realistische Schwellwerte hängen von der durchschnittlichen Anzahl der Scheitelpunkte pro Merkmalsinstanz sowie der Auflösung des Dreiecksnetz und den verwendeten Algorithmen zur Generierung oder Netzunterteilung ab. Zur Berechnung der finalen Charakteristiken der einzelnen Instanzen wird für jede dieser zuerst die konvexe Hülle berechnet und die Koordinaten der umschließenden Bounding-Box. Dann wird auf Basis der konvexen Hülle der Volumenschwerpunkt, eine Approximation des Flächenschwerpunkts sowie der Mittelwert der Instanz ermittelt. Da alle Dreiecksnetze des behandelten Datensatzes 2-Mannigfaltgikeiten sind und die Oberfläche somit in sich geschlossen ist, werden die auf den Clustern basierenden Metriken leicht beeinflusst. Dieser Effekt ist jedoch vernachlässigbar und wird zusätzlich mit steigender Anzahl an Scheitelpunkten sehr klein. 

## **5.3.6 Berechnung der Metriken** 

Der Informationsgehalt der reinen Vorhersagen der neuronalen Netze ist zwar sehr hoch, jedoch ohne Nachverarbeitung für die Montage und Verdrahtung der Schaltschrankkomponenten von geringer Bedeutung. Basierend auf den Segmentierungen werden deshalb die Position, Größe und Normalvektoren der segmentierten Merkmale berechnet. Als Beispiel wird hierfür eine durch das DiffusionNet segmentierte Kabeleinführung in Abbildung 44 betrachtet. Die Position wird in dreifacher Ausführung über verschiedene Schwerpunkte dargestellt. Zum einen wird der Clusterschwerpunkt (gelb) berechnet, welcher die Mitte der Scheitelpunkte des segmentierten Dreiecksnetzes eines Merkmals darstellt. Dieser ist zwar eine gute Annäherung kann jedoch unter bestimmten Umständen eine Tendenz in eine Richtung mit kleineren Dreiecken aufweisen, da dann die Dichte der Scheitelpunkte lokal höher ist. Dieser Effekt ist in (c) sichtbar, da der ClusterSchwerpunkt in Richtung der dichteren Mesh-Struktur nach rechts verschoben ist. Obwohl Oberflächendreiecksnetze kein Volumen haben, kann bei stark gekrümmten Oberflächen dennoch der Volumenschwerpunkt berechnet werden, indem zuerst die konvexe Hülle kalkuliert und von dieser dann der Volumenschwerpunkt berechnet wird. Dies hat den Vorteil, dass der Einfluss der Mesh-Struktur auf den Schwerpunkt minimiert wird. Der Cluster-Schwerpunkt (rot) stellt, wie in (c) illustriert, die tatsächliche Mitte besser dar. Zuletzt ist der Flächenschwerpunkt insbesondere bei Kontaktierungen und Kabeleinführungen interessant, da dadurch die Tiefe der Aussparung bestimmt werden kann. Der Flächenschwerpunkt (grün) wird ermittelt, indem die äußersten Randscheitelpunkte iterativ eliminiert werden. Dies wird so lange wiederholt, bis entweder nur noch Randscheitelpunkte vorhanden sind oder 

76 

5 Datenaggregation, Modellierung und Technische Implementierung 

das Dreiecksnetz aus nur noch einem einzigen Punkt besteht, welcher den Flächenschwerpunkt darstellt. Bei genauer Betrachtung fällt auf, dass dies nicht der Flächenschwerpunkt der Mannigfaltigkeit, sondern der zugrundenliegenden MeshStruktur ist. Deshalb wird vor Anwenden dieses Verfahrens das Dreiecksnetz über die Smooth-Subdivision gleichmäßig unterteilt, wodurch dieser Effekt minimiert wird [88]. Aus (a-c) wird dennoch deutlich, dass dieser Punkt auch nach der SmoothSubdivision noch vom tatsächlichen Oberflächenschwerpunkt abweichen kann. 

**==> picture [316 x 129] intentionally omitted <==**

**----- Start of picture text -----**<br>
𝑣 𝑜 −𝑣 𝑠<br>𝑣 𝑜<br>𝑣 𝑠<br>(a) 0 (b) (c)<br>**----- End of picture text -----**<br>


**==> picture [13 x 10] intentionally omitted <==**

**----- Start of picture text -----**<br>
(d)<br>**----- End of picture text -----**<br>


_Abbildung 44: Cluster-, Volumen-, Oberflächenschwerpunkt, sowie Normalvektor einer mittels des DiffusionNet segmentierten Kabeleinführung_ 

Ergänzend zum Flächenschwerpunkt wird deshalb zuletzt noch der Mittelpunkt aller Randscheitelpunkte berechnet (d), welcher nahe dem Mittelpunkt der Öffnung (dunkelblau) einer Kabeleinführung oder Kontaktierung liegt. Durch die Differenz des 

77 

5 Datenaggregation, Modellierung und Technische Implementierung 

Vektors 𝑣 𝑜 zum Mittelpunkt der Randscheitelpunkte und des Vektors des Flächenschwerpunkts 𝑣 𝑠 kann der Normalvektor des Merkmals bestimmt werden. Aufgrund der approximativen Bestimmung des Flächenschwerpunktes kann dieser Normalvektor ebenfalls ungenau sein. Um den Normalvektor in die korrekte Richtung senkrecht zur mittleren Fläche auszurichten wird der Vektor zu zwei Seiten der Bounding-Box des Merkmals parallel ausgerichtet. Dadurch resultiert automatisch ein orthogonaler Vektor ⃗ in der Mitte des Features. Die Korrektur und Ausrichtung des Normalvektors finden dabei in diejenigen beiden Richtungen statt, deren Winkel zum Vektor minimal sind. Nach der Anpassung der Ausrichtung wird der Vektor in die Mitte der Bounding-Box gesetzt und zeigt letztlich vom unteren zum oberen orangen Punkt. Abschließend wird die Größe jedes Features über die Oberfläche bestimmt. Die Gesamtoberfläche setzt sich aus der Summe der Oberflächen aller Dreiecke zusammen. 

## **5.3.7 Inferenzpipeline** 

Aus den einzelnen beschriebenen Bestandteilen lässt sich die gesamte Inferenzpipeline zusammensetzen. Diese ist in Abbildung 45 dargestellt. Die empfohlenen Komponenten werden dabei durch schwarze Pfeile dargestellt. Mögliche erprobte Alternativen sind grau dargestellt und müssen nicht zusätzlich verwendet werden. Ausgehend von einer STEP-Datei wird diese zunächst konvertiert. Hier empfiehlt sich die Verwendung von OCCT, da diese Bibliothek sowohl die Konvertierung der STEP-Dateien in STL-Dateien durchführen kann als auch das Modell aus verschiedenen Winkeln rendern und Bilder aufnehmen kann. Die Alternative bzgl. der Dateikonvertierung ist gmsh mittels trimesh. Die Bilder können alternativ noch mittels FreeCAD erstellt werden, welche allerdings weniger hochauflösend sind. Zudem können dort die Bilder nicht an die Anzahl der Pixel angepasst werden, weshalb eine Nachverarbeitung erforderlich wäre. STL-Dateien können über viele verschiedene Bibliotheken in OBJ-Dateien konvertiert werden, weshalb hier kein bevorzugtes Verfahren aufgelistet ist. Im Rahmen der Arbeit wurde die Konvertierung mittels Open3D durchgeführt. Die anschließende Skalierung auf eine annähernd einheitliche Anzahl an Scheitelpunkten wird mittels Manifold und ManifoldPlus durchgeführt [85][86]. Die Einbettung der beiden Paper stellt die einzige sinnvolle Methode zur Erzeugung fehlerfreier Oberflächen von Mannigfaltigkeiten dar. Die Erzeugung fehlerfreier Mannigfaltigkeiten durch differentielles Rendering hat keine Erfolge gezeigt und ist durch die benötigte Dauer zur Erzeugung einer fehlerfreien Oberfläche ungeeignet. Ebenfalls stellen stark konkave Dreiecksnetze hierfür ebenfalls eine Hürde dar. Bei Verwendung des MeshCNN kann die Inferenz auf Basis der OBJ-Dateien erfolgen. Für das DiffusionNet empfiehlt sich dagegen die Konvertierung in OFF-Dateien. Die Konvertierung erfolgt mittels einer eigenen performanten Implementierung über die Python C API. 

78 

5 Datenaggregation, Modellierung und Technische Implementierung 

**==> picture [671 x 386] intentionally omitted <==**

**----- Start of picture text -----**<br>
gmsh/trimesh<br>.stl .obj Manifold(Plus)<br>STEP-Modell OCC-Core<br>FreeCAD .obj mit x<br>.off meshparser<br>Scheitelpunkten<br>! 512×512 Bilder I1 :<br>I1 DiffusionNet MeshCNN<br>ti<br>|I bottom.png front.png left.png 290 290<br>I I1 rds®, Lye7 TN ryie, Weyaac]<br>I!L ee top.png rear.png ee right.png i;I ieraStoreKWAY'1PONSAENOCXONEv NeSINbeANASSVAetKtaya yeeoeporero1PONSAPONSAOXKARORSKTMStnbeANYanteeKaNavtSY<br>6-000 O--000<br>Qe .vy /aNa aay “1 .ys /wya awe“1<br>We ‘vig: ay > ‘> /<br>YOLOv6 . PdaVita) ie1 ammm Bow1 7)lySao VOM1IMry SYNvN V1ETSfee VOM1LIMESNGAfe<br>eco S C9): : LED»ew oN I LANNYVe MT VeVe ANATVE<br>Ke SAY > i k| AZ ve SM Ve Su<br>PSPSS fryeren ‘legNG a \. LM y \/ NRG\\ Ss I eco eco<br>PONSbONP KENeryUA asSS ee \ ar L$ela v4| ee1 |<br>BOXONSNTNae i} Ne J > \ : <a<br>ioaan hyats\n | i : Za"x . fifi it I I<br>ie noo tt ye \ . ° —<br>VaVS vor “uti U Ksy ANe I \!<br>vos DA 1 )>x | if<br>\ Kot™“ 1i$ i _s re Sn“ — | I<br>\ Mes Me To i ag YG J<br>eee 5 i a el |<br>YWrVZ =Ex SEVS =ve GN—_* :<br>jorote eee  e - e<br>ooo 3D-Bounding-<br>3D-Projektion Box F Cluster-, Volumen- und Flächenschwerpunkte<br>Open3D Open3D<br>**----- End of picture text -----**<br>


## _Abbildung 45: Gesamtinferenz der implementierten Netzwerke_ 

79 

5 Datenaggregation, Modellierung und Technische Implementierung 

Nach der Inferenz über das DiffusionNet oder MeshCNN muss der Output nachverarbeitet werden. Im Falle des DiffusionNets liegt ein Tensor der Größe (Scheitelpunkte, 5) vor. Die zweite Dimension stellt dabei je Scheitelpunkt eine Wahrscheinlichkeitsverteilung über alle Klassen dar, wobei die höchste Wahrscheinlichkeit verwendet wird. Daraus resultiert ein eindimensionaler Tensor mit identischer Größe zur ersten Dimension des Scheitelpunkttensors, wodurch eine eindeutige Zuordnung auf Scheitelpunktebene möglich ist. Mittels Open3D werden die Segmentierungsmasken je Klasse verwendet, um einzelne Instanzen zu identifizieren und darauf basierend die Schwerpunkte zu berechnen. Im Falle des MeshCNN liegt der Output in Form einer Wahrscheinlichkeitsverteilung über alle Klassen vor, welche allerdings den Kanten zugeordnet werden. Die Nachverarbeitung wurde hierfür nur teilweise zur Analyse der Ergebnisse implementiert, da die Ergebnisse weniger vielversprechend sind. Dieses Netzwerk ist deshalb nicht für eine e2e-Inferenz geeignet. Unabhängig von der Inferenz auf den Dreiecksnetzen wird ein weiteres CNN auf Basis von YOLOv6 verwendet. Der Input liegt in Form von sechs Bildern der Größe 512×512 Pixel vor. Nach dem Durchlaufen des Netzwerks liegen dann die Koordinaten der identifizierten Bounding-Boxen je Klasse vor. Abhängig von der Gewissheit je Bounding-Box können diese über einen Schwellwert gefiltert werden, da Inferenzen mit sehr geringer Gewissheit keine Aussagekraft aufweisen. Die verbleibenden Bounding-Boxes werden anschließend auf das 3D-Objekt projiziert. Dies erfolgt über das Mapping der Bounding-Boxen auf die entsprechenden Seitenflächen. Die mögliche Drehung von Komponenten im Raum hat hierauf keinen Einfluss, da diese sowohl beim Aufnehmen der Bilder als auch in Open3D identisch ist, weil diese nicht bei der Konvertierung von STEPModell zu Dreiecksnetz verloren geht. Da nach der Abbildung der Bounding-Boxen auf das Dreiecksnetz eine Dimension fehlt, wird diese ergänzt. Die hinzugefügten Koordinaten der fehlenden Dimension werden auf die volle Länge der Komponenten in Richtung der Dimension gesetzt. Dadurch wird die 2D-Bounding-Box in eine 3DBounding-Box überführt, indem diese in die fehlende Dimension gezogen wird. Der Teilbereich des Dreiecksnetzes, welcher sich in der 3D-Bounding-Box befindet, kann dann ausgeschnitten und separat weiterverarbeitet werden. Identisch zu beiden vorherigen Ansätzen liegt nun eine Segmentierung der Scheitelpunkte vor, auf welcher entsprechende Schwerpunkte berechnet werden können. Nach der Berechnung der finalen Vektoren können diese zur Validierung visualisiert oder als Input für weitere Berechnungen genutzt werden. Nachfolgend werden in Abbildung 46 die einzelnen Schritte zur Abbildung einer 2D-Bounding-Box in den 3D-Raum visualisiert. Nach der Inferenz des Ausgangsbildes (a) erfolgt das Ausschneiden des Bildes (b), um eine identische Größe zur 3D-Bounding-Box zu erzeugen. In (c) werden die 2D-Bounding-Boxen aus (b) in den 3D-Raum abgebildet und um die fehlende Dimension erweitert. 

80 

5 Datenaggregation, Modellierung und Technische Implementierung 

**==> picture [378 x 11] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a) Ausgangsbild (b) Inferiert + ausgeschnitten (c) Projektion<br>**----- End of picture text -----**<br>


## _Abbildung 46: Prozess zur Abbildung einer 2D-Bounding-Box in den 3D-Raum_ 

## **5.4 Optimierung der neuronalen Netze** 

Der Erfolg beim Trainieren neuronaler Netze ist durch eine Vielzahl verschiedener Hyperparameter bedingt. Vor dem langfristigen Trainieren einzelner Modelle empfiehlt sich deshalb die Ermittlung geeigneter Hyperparameter. 

## **5.4.1 Hyperparameteroptimierung** 

Für die Implementierung des abgewandelten MeshCNN wurden die jeweils empfohlenen Hyperparameter aus der Literatur entnommen [61][60]. Das YOLOv6CNN wurde ebenfalls mittels den empfohlenen Hyperparametern trainiert [98]. Die HPO hat demnach insbesondere für das DiffusionNet stattgefunden. Dabei wurden die in Tabelle 3 aufgelisteten Hyperparameter in den entsprechenden Intervallen oder nach den gelisteten Kategorien optimiert. 

_Tabelle 3: HPO des DiffusionNet_ 

|Hyperparameter|Art|Werte|
|---|---|---|
|Eingangsmerkmal|kategorisch|XYZ, HKS|
|Lernrate|logarithmisch|[1e-5, 1e-1]|
|Abklingfrequenz|kategorisch|25, 50, 100, 250, 500|
|Abklingrate|uniform|(0, 1)|
|Verlustfunktion|kategorisch|NLL, CE|
|Diffusionsblöcke|kategorisch|1, 2, 3, 4, 5|



81 

5 Datenaggregation, Modellierung und Technische Implementierung 

|Hyperparameter|Art|Werte|
|---|---|---|
|Diffusionsblock Dimension|kategorisch|32, 64, 128, 256, 512, 1024|
|Anzahl Eigenvektoren|kategorisch|32, 64, 128, 256, 512, 1024|
|Dropout|uniform|(0, 1)|



## **5.4.2 Ray Tune** 

Aufgrund immer komplexerer Modelle, welche auf einzelnen GPUs zeitintensiv über Stunden, Tage oder Wochen trainiert werden, ist der Bedarf für verteiltes Erproben des Hyperparameter-Spektrums ebenfalls relevant. Im Rahmen dieser Arbeit wurde deshalb Ray Tune verwendet, um unterschiedliche Hyperparameter parallel statt sequenziell zu ermitteln. Der Grad der Parallelität hängt dabei maßgeblich vom verfügbaren Grafikspeicher und der Größe des Modells ab. [35] Ray Tune bietet sowohl eine Nutzer-API für das Training neuronaler Netze sowie eine Planungs-API zum Zwecke der Forschung an Modellsuchalgorithmen selbst an. Da Trainingsskripte in der Regel als Schleife über mehrere Optimierungsschritte implementiert sind, erfolgt die Protokollierung der Zwischenergebnisse über die Mittelung der Zwischenergebnisse pro Epoche. Zur Unterstützung des gesamten Spektrums der Suchalgorithmen wird Ray Tune Zugang zu diesen Zwischenergebnissen und zu Momentaufnahmen des Trainingszustandes gewährt. Darüber hinaus kann Ray Tune die Hyperparameter während des Trainings manipulieren. Neben verschiedenen Suchalgorithmen besteht zudem die Möglichkeit Versuchsreihen zu erstellen und diese in ihrer Ausführung zu kontrollieren. Daraus ergibt sich die Frage einer geeigneten Versuchsreihenplanung. Ray Tune stellt verschiedene Versuchsplaner zur Verfügung, wobei diese die Versuche unter Berücksichtigung der verfügbaren Ressourcen ausführen. Sehr einfache Versuchsplaner führen die Versuche so lange aus, bis eine Stoppbedingung erfüllt ist. Im Falle nicht ausreichend zur Verfügung stehenden Ressourcen im Rechencluster erfolgt die Ausführung sequenziell. Versuchsplaner ermöglichen darüber hinaus noch weitere automatisierte Mechanismen: [35] 

- Vorzeitige Beendigung aufgrund nicht erfolgreicher Zwischenergebnisse 

- Anpassen des Veränderungsgrads von Hyperparametern 

- Kopieren von Hyperparametern erfolgreicher Studien und Erkundung nahegelegener Hyperparameterräume 

- Abfrage einer gemeinsamen Datenbank zur Auswahl vielversprechender Hyperparameter 

- Priorisierung der Zuweisung von Ressourcen bei hoher Versuchsanzahl 

82 

5 Datenaggregation, Modellierung und Technische Implementierung 

Die Wahl dieser Optimierungsbibliothek ist zum einen durch die aufgezählten Funktionen und zum anderen durch die einfache Integration in DL-Frameworks begründet. [35] 

83 

6 Ergebnisse und Diskussion 

## **6 Ergebnisse und Diskussion** 

Im Rahmen dieses Kapitels wird in Anlehnung an die DMME die Umsetzung des Evaluationsschrittes beschrieben. Im ersten Teilkapitel wird der Datensatz selbst untersucht, um ein allgemeines Verständnis für diesen aufzubauen. Die Datenexploration untergliedert sich dabei zum einen in eine Analyse aller aggregierten Daten und zum anderen in eine Analyse der für das Training der MLModelle verwendeten Daten. Anschließend wird die Auswahl der Trainingsdaten kurz beschrieben. Das zweite Unterkapitel beschäftigt sich mit den erzielten Ergebnissen durch die im vorherigen Kapitel implementierten ML-Modelle sowie geeigneten Bewertungskriterien. Abschließend werden Nutzen und Limitationen der verwendeten Modellarchitekturen diskutiert. 

## **6.1 Datenexploration und Datenverständnis** 

Für eine sinnvolle Interpretation der Ergebnisse dieses Kapitels, wird detailliertes Wissen über die verwendeten Daten benötigt. Im Rahmen der Arbeit wurden insgesamt ca. 200 GB an Rohdaten aggregiert. Da nur ein Bruchteil dieser für das Training der ML-Modelle verwendet wurde, werden alle Daten und die gelabelten Daten separat analysiert. Im Folgenden werden die einzelnen Segmentierungsklassen als Merkmal oder einfach als Klasse bezeichnet. Eine einzelne Ausprägung wie beispielsweise eine Kontaktierung oder eine Beschriftungsfläche wird als Merkmalsinstanz oder Merkmalsausprägung bezeichnet. 

## **6.1.1 Strukturierte Analyse der Rohdaten** 

Die Rohdaten liegen als STEP-Dateien vor. Insgesamt umfasst der Datensatz 46068 Dateien, welche von 75 verschiedenen Herstellern stammen. Da viele Komponenten jeweils eines Herstellers geometrisch identisch sind und im Rahmen dieser Arbeit lediglich eines dieser Duplikate relevant ist, wurden alle geometrisch einzigartigen Dateien zuerst in STL- und anschließend in OBJ-Dateien konvertiert. Die Verteilung der Häufigkeiten der einzelnen Dateiformate je Hersteller ist in Abbildung 47 dargestellt. Die genauen Häufigkeiten sind Anhang B – Datensatzinformationen zu entnehmen. Da die Konvertierung nicht aller STL-Dateien fehlerfrei bzw. in einem angemessenen Zeitrahmen erfolgt, ist die Anzahl von OBJ-Dateien geringer. Es wurde bei der Konvertierung dabei ein maximales zeitliches Limit von 300 Sekunden gesetzt. Dieses Limit ist aufgrund der sehr hohen Varianz in der Größe der Dateien notwendig. 

84 

6 Ergebnisse und Diskussion 

**==> picture [454 x 631] intentionally omitted <==**

**----- Start of picture text -----**<br>
|||||||
|---|---|---|---|---|---|
|Wiesemann & Theis|
|Wieland|
|Wenglor|
|Weintek|
|Weidmueller|
|Wago|
|Unitronics|
|Turck|
|Trafomodern|
|Tempa Pano|
|Tecnotion|
|Sun Hydraulics|
|Stoeber|
|Stasto Automation|
|Socomec|
|SMC|
|Sigmatek|
|Siemens|
|SEW|
|Schneider Electric|
|Schmersal|
|Puls|
|PR Electronics|
|Pilz|
|Phoenix|
|Omron|
|OBO Bettermann|
|Novatek-Electro|
|Murrelektronik|
|Mitsubishi|
|Mean Well|
|Mastervolt|
|LS Industrial Systems|
|Lovato Electric|
|Lenze|
|Kunbus|
|Kistler|
|Keyence|
|KEB|
|Kaleja|
|Jean Müller|
|Janitza|
|Igus|
|IFM|
|IDEC|
|IAI|
|Hensel|
|Häwa|
|Grundfos|
|Finder|
|Eurotherm|
|ETI|
|ETA|
|Elitra|
|Eldon|
|Eaton|
|EAE|
|Dina Elektronik|
|Dehn|
|Danfoss|
|Crouzet|
|ComAp|
|Cliff|
|Carlo Gavazzi|
|Camtec Power Supplies|
|Bulgin|
|Brainboxes|
|Bosch Rexroth|
|Bopla|
|Bicker Elektronik|
|Beckhoff|
|Balluff|
|B&R|
|Allen Bradley|
|ABB|
|1|10|100|1000|10000|100000|
|OBJ|STL|Duplikate|STP|

**----- End of picture text -----**<br>


_Abbildung 47: Häufigkeiten der verschiedenen Dateiformate nach den Konvertierungen je Hersteller_ 

85 

6 Ergebnisse und Diskussion 

Abbildung 48 zeigt, dass der Großteil der Daten zwar in angemessenen Dateigrößen vorliegt, jedoch einige STEP-Dateien bis zu 115 Megabyte (MB) groß sind. Da eine STL-Datei wesentlich speicherintensiver als eine STEP-Datei ist, sind letztere bis zu 600 MB in diesem Datensatz groß. 

**==> picture [454 x 227] intentionally omitted <==**

**----- Start of picture text -----**<br>
1.00E+05<br>1.00E+04<br>1.00E+03<br>1.00E+02<br>1.00E+01<br>1.00E+00<br>[0,5) [25,30) [50,55) [75,80) [100,105)<br>Dateigrößenintervalle in MB →<br>Anzahl STEP-Dateien →<br>**----- End of picture text -----**<br>


## _Abbildung 48: Histogramm der Dateigrößen aller STEP-Dateien des Datensatzes_ 

Dateien dieser Größe sind deshalb nicht zum Zwecke dieser Arbeit sinnvoll, da während des Downsamplings auf 6000 Scheitelpunkte, sonst zu viele Informationen verloren gehen würden. Auffällig ist zudem, dass sehr große Dateien nicht zwangsläufig geometrisch komplexer aufgebaut sind, sondern lediglich unwichtige Herstellerinformationen in die STEP-Datei modelliert wurden. Nach der Konvertierung in eine STL-Datei wurden alle Duplikate eliminiert. Insgesamt sind 54,12 % aller STEP-Dateien Duplikate. Daraus resultieren 21134 geometrisch verschiedene STEP-Dateien im Datensatz. Die Anzahl an Unikaten sowie die Konvertierungsrate je Hersteller ist Tabelle B.1 des Anhangs zu entnehmen. Nach der Konvertierung in STL-Dateien wurden diese nach Kategorien gefiltert, sodass insgesamt 4729 Dateien im OBJ-Format vorliegen. Die gefilterten Kategorien wurden dabei so gewählt, dass lediglich Komponenten ausgeschlossen werden, welche definitiv irrelevant sind. Die ignorierten Kategorien sind Tabelle B.3 des Anhangs zu entnehmen. Da die OBJ-Dateien eine stark verschiedene Anzahl an Scheitelpunkten aufweisen, wurden diese anschließend skaliert. Vor der Skalierung lag das Minimum der Anzahl an Scheitelpunkten bei 200, das Maximum bei 1420817 und der Mittelwert bei ca. 30483. Nach der in Abbildung 49 dargestellten Skalierung von vvor nach vnach liegt das Minimum bei 5542, das Maximum bei 6002 und der Mittelwert bei ca. 5972 Scheitelpunkten. Es wird hierbei die Anzahl an Scheitelpunkten vor und nach der Skalierung für jede der 4729 Dateien dargestellt. Die Anzahl der 

86 

6 Ergebnisse und Diskussion 

Scheitelpunkte unterscheidet sich auch nach der Skalierung noch geringfügig, da die Dreiecksnetze auf 12000 Dreiecke skaliert wurden. Aufgrund der Euler-Charakteristik und der spezifischen Beschaffenheit jedes einzelnen Dreiecksnetzes ist eine Skalierung auf eine exakte Anzahl deshalb nicht immer möglich. Zusätzlich zur Skalierung wurden für Graph- und Tensoroperationen wichtige Eigenschaften der 2- Mannigfaltigkeiten mittels [85] sichergestellt. 

**==> picture [454 x 227] intentionally omitted <==**

**----- Start of picture text -----**<br>
6100<br>6000<br>5900<br>5800<br>5700<br>5600<br>5500<br>1.00E+02 1.00E+03 1.00E+04 1.00E+05 1.00E+06 1.00E+07<br>vvor →<br>OBJ<br> →<br>nach<br>v<br>**----- End of picture text -----**<br>


_Abbildung 49: Skalierung der OBJ-Dateien auf eine fast einheitliche Anzahl an Scheitelpunkten_ 

## **6.1.2 Analyse des Datensatzes** 

Von den 4729 konvertierten und aufbereiteten OBJ-Dateien wurden anschließend 234 Dieser Dateien gelabelt. Dieses Kapitel betrachtet ausschließlich die verbleibenden 234 Dateien. Nach dem Filtern der Dateien erfolgte eine manuelle Inspizierung aller verbleibenden Dateien. Im Rahmen dieser Arbeit wurden bewusst neben Reihenklemmen auch Relais und verschiedene weitere Komponenten ausgewählt, um das grundlegende Konzept zu erproben sowie bessere Generalisierung zu erzielen. Wie auch aus Abbildung 47 hervorgeht, ist der Komponentenmarkt äußerst divers jedoch dennoch durch wenige große Unternehmen dominiert. Siemens und Phoenix stellen die mit Abstand meisten verschiedenen Komponenten her. Entgegen der hohen Anzahl an Dateien von Siemens stellt Phoenix die meisten verschiedenen Komponenten her, weshalb bewusst auch von dieser Firma am meisten Dateien im gelabelten Datensatz vorhanden sind. Abbildung 50 stellt die Verteilung der Hersteller im finalen Datensatz dar. Neben dem Großteil der Dateien von Phoenix, Wago und Siemens wurden bewusst auch Komponenten kleinerer Firmen gelabelt, um die ML-Modelle 

87 

6 Ergebnisse und Diskussion 

schließlich auf eine gute Generalisierung zu prüfen. Diese machen ca. 19 % des Datensatzes aus. 

**==> picture [454 x 227] intentionally omitted <==**

**----- Start of picture text -----**<br>
ABB<br>Allen Bradley<br>Beckhoff<br>Carlo Gavazzi<br>Dehn<br>Phoenix<br>Siemens<br>Wago<br>Weidmueller<br>**----- End of picture text -----**<br>


## _Abbildung 50: Verteilung der Hersteller des gelabelten Datensatzes_ 

Abgesehen von den Herstellern wurde zudem bewusst auf die Geometrie der ausgewählten Komponenten geachtet. Aufgrund der Ähnlichkeit und nur minimaler Unterschiede einiger Bauteile, wurden während der Auswahl der Komponenten einige Kriterien berücksichtigt, um das Überanpassen auf ein Bauteil zu verhindern. Es wurden nur Bauteile ausgewählt, welche eines der folgenden Kriterien erfüllen: 

- Bauteile mit verschiedener Breite, Höhe oder Länge 

- Bauteile mit unterschiedlichen Anzahlen an Kontaktierungen, Aufrastpunkten, Kabeleinführungen oder Beschriftungsflächen 

- Bauteile mit vom Standard abweichenden Merkmalsausprägungen, wie z.B. schräge Kontaktierungen 

- Bauteile mit stark asymmetrischen Merkmalen 

Unter der Berücksichtigung dieser Kriterien wurde versucht eine Tendenz in den Daten weitestgehend zu vermeiden. Einzelne Bauteile mit stark asymmetrischen Merkmalen sollen hier beispielsweise helfen das Erlernen von Symmetrieeigenschaften der Merkmale zu vermeiden, da dieses zwar häufig gegeben ist, jedoch keine Bedeutung für die Segmentierung hat. Durch die Variation der Anzahl an Merkmalsausprägungen wird versucht das Erlernen einer Korrelation zwischen den Merkmalsausprägungen untereinander möglichst zu vermeiden. Dieses ist ebenfalls oftmals vorhanden, jedoch nicht zwangsläufig notwendig. Ein Beispiel hierfür sind Beschriftungsflächen in der Nähe einer Kontaktierung. Diese sind oftmals in der Nähe angebracht, können jedoch fast jede beliebige Position in der Komponente annehmen. Auf Basis der Labels wurden die Merkmalshäufigkeiten 

88 

6 Ergebnisse und Diskussion 

je Bauteil durch die Graphkonnektivität berechnet. Dies ist kein exakter Wert der tatsächlichen Merkmalsausprägungen, sondern lediglich eine minimale Schätzung dieser jedoch trotzdem für das Verständnis der Verteilung hilfreich, da die Inferenz schlussendlich auch Gebrauch dieser Technik macht. Ebenfalls wird die Anzahl dann zu hoch berechnet, falls vereinzelte Scheitelpunkte falsch gelabelt wurden, was insbesondere bei sehr komplexen Geometrien nicht ausgeschlossen ist. Abbildung 51 stellt die Anzahl der Ausprägungen für alle vier Merkmale dar. Das Gehäuse wird hier und in folgenden Abbildungen teilweise bewusst außen hervorgelassen, da es ohnehin nur eine zusammenhängende Instanz gibt und immer aus den restlichen Scheitelpunkten besteht. Es dient daher als Standardklasse oder Hintergrund der eigentlichen Merkmale. Der Datensatz wurde mittels eines 70/20/10 Train/Val/TestSplits aufgeteilt. Abbildung 51 zeigt, dass die Testdaten ungefähr der gleichen Verteilung der Trainings- und Validierungsdaten folgen. Dies ist wünschenswert, um das Training des Modells auf realitätsnahen Daten zu gewährleisten. 

_Abbildung 51: Verteilungen der Instanzen der vier Merkmale Kontaktierung, Aufrastpunkt, Kabeleinführung und Beschriftungsfläche für den Trainings- und Testdatensatz_ 

Nicht nur die Anzahl der Merkmalsinstanzen pro Komponente ist höchst variabel, sondern ebenfalls die Anzahl an Labels einer Klasse. Die Anzahl an Scheitelpunkten hängt zum einen von der Anzahl der Merkmalsinstanzen und zum anderen von der Komplexität einer Instanz selbst ab, da die Darstellung einfacher Geometrien mit weniger Scheitelpunkten möglich ist. Abbildung 52 zeigt, dass den durchschnittlich 

89 

6 Ergebnisse und Diskussion 

größten Anteil der Labels, abgesehen vom Gehäuse, die Kontaktierungen ausmachen, gefolgt von gleichermaßen Kabeleinführungen und Beschriftungsflächen. 

**==> picture [454 x 284] intentionally omitted <==**

**----- Start of picture text -----**<br>
120<br>100<br>80<br>60<br>40<br>20<br>0<br>0-99 500-599 1000-1099 1500-1599 2000-2099<br>Anzahl der Scheitelpunkte je Merkmal →<br>Kontaktierung Aufrastpunkt Kabeleinführung Beschriftungsfläche<br>**----- End of picture text -----**<br>


_Abbildung 52: Histogramm der Häufigkeiten der Anzahl an Scheitelpunkten je Merkmal_ 

Aufgrund der sehr einfachen Geometrie und geringen Größe der Aufrastpunkte sind diese am wenigsten stark vertreten und können meist mit 100 bis 200 Scheitelpunkten dargestellt werden. Der prozentuale Anteil der einzelnen Labels ist für die Merkmale stark unausgewogen: 

- Gehäuse: 70,7 % 

- Kontaktierung: 14,32 % 

- Aufrastpunkt: 2,83 % 

- Kabeleinführung: 5,74 % 

- Beschriftungsfläche: 6,41 % 

Dieses starke Ungleichgewicht legt grundsätzlich gewichtete Verlustfunktionen für diesen Datensatz nahe. Aus den Instanzen je Merkmal sowie der Gesamtanzahl der Labels aus beiden vorherigen Abbildungen ergeben sich die Anzahlen der Häufigkeiten der Labels je Klasse. Abbildung 53 stellt die Verteilungen entsprechend dar. Diese Verteilungen geben einen Überblick wie komplex einzelne Merkmalsinstanzen sind. Zudem stellt der Boxplot Richtwerte für die Mindestanzahl an Scheitelpunkten pro Merkmalsinstanz dar. Ein sinnvoller Richtwert ist hier das untere 

90 

6 Ergebnisse und Diskussion 

Quartil. Die unteren Quartile liegen für die einzelnen Merkmale entsprechend der Reihenfolge in Abbildung 53 bei 58,25, 21, 35 sowie 117,5. 

## _Abbildung 53: Verteilungen der Scheitelpunktanzahl pro Merkmalsinstanz_ 

Sämtliche Überlegungen dieses Datensatzes sind nur bedingt auf Hochskalierungen dieses Datensatzes gültig. Während bei der einfachen Netzunterteilung die Verhältnisse identisch bleiben und folglich die Gewichte für die Verlustfunktion nicht beeinflusst werden, ist eine Neuberechnung dieser bei der glatten Netzunterteilung notwendig, da nicht alle Bereiche des Dreiecksnetzes gleichermaßen vereinfacht werden. 

## **6.2 Auswertung** 

Dieses Kapitel wertet die Ergebnisse der implementierten neuronalen Netze aus. Hierfür wird zunächst auf die verwendeten Hyperparameter eingegangen sowie anschließend die Kennwerte der optimierten Netze diskutiert. 

## **6.2.1 Analyse der optimierten Hyperparameter** 

Die HPO wurde mittels Ray Tune und einem Asynchronous Successive Halving Algorithm (ASHA) durchgeführt. Dieser Planungsalgorithmus führt aggressives EarlyStopping unter sehr hoher Parallelität durch. Der ASHA-Planer wurde aufgrund empirischer Belege für dessen Effizienz verwendet. ASHA übertrifft moderne HPO- 

91 

6 Ergebnisse und Diskussion 

Algorithmen und skaliert linear mit der Anzahl an Recheninstanzen und ist daher für massive Parallelität geeignet. Trotz nur einer einzigen GPU können dennoch CPUBerechnungen auf allen Kernen parallelisiert werden. [102] Tabelle 4 listet die verwendeten Hyperparameter für das DiffusionNet auf. Unabhängig vom erprobten DNN wurden die Gewichte für die Verlustfunktion aus Tabelle 4 verwendet. 

_Tabelle 4: Gewichte für die verwendeten Verlustfunktionen_ 

|Klasse|Wert|
|---|---|
|Gehäuse|46,4605|
|Kontaktierung|221,5453|
|Aufrastpunkt|1156,4556|
|Kabeleinführung|546,8523|
|Beschriftungsfläche|514,3766|



## **DiffusionNet** 

Insbesondere komplexere DiffusionNets mit mehr als drei Diffusionsblöcken sowie Berechnungen von mehr als 64 Eigenvektoren und geringere Dropoutraten ermöglichen dem Netz komplexere Funktionen zu approximieren und haben zu Overfitting geführt. Tabelle 5 listet die verwendeten Hyperparameter des besten DiffusionNets auf. 

_Tabelle 5: Hyperparameter des besten DiffusionNets_ 

|Hyperparameter|Art|Wert|
|---|---|---|
|Eingangsmerkmal|kategorisch|XYZ|
|Lernrate|logarithmisch|1e-3|
|Abklingfrequenz|kategorisch|100|
|Abklingrate|uniform|0,75|
|Verlustfunktion|kategorisch|NLL|
|Diffusionsblöcke|kategorisch|3|
|Diffusionsblock Dimension|kategorisch|64|
|Anzahl Eigenvektoren|kategorisch|64|
|Dropout|uniform|0,3|



92 

6 Ergebnisse und Diskussion 

## **6.2.2 Kennwerte der optimierten Netze** 

Zur Segmentierung der Dreiecksnetze in die fünf Kategorien wurden zwei grundlegend verschiedene DNNs über das DiffusionNet und das MeshCNN trainiert. Letzteres wurde in Anlehnung an das MedMeshCNN abgewandelt. Segmentierungsaufgaben erfordern neben der Genauigkeit noch weitere Metriken zur Beurteilung, da sonst bei einer falschen Interpretation der Genauigkeit falsche Schlüsse gezogen werden. Beide DNNs werden deshalb ergänzend zum Loss auch über den DiceKoeffizienten sowie den Jaccard-Index bewertet. Alle beispielhaften Abbildungen der Segmentierung werden anhand einer Gegenüberstellung der Labels und Vorhersagen visualisiert, wobei die Labels jeweils links und die Vorhersagen rechts davon abgebildet sind. 

## **DiffusionNet** 

Beim DiffusionNet wurde der NLL-Loss verwendet. Dieser nimmt bei einer perfekten Segmentierung einen Wert von null an, weshalb das Ziel in der Minimierung bis zu diesem Wert besteht. Abbildung 54 stellt den Trainings- und Validierungs-Loss gegenüber. Deutlich wird hierbei, dass beide Kurven ab ca. Epoche 120 divergieren. 

**==> picture [454 x 284] intentionally omitted <==**

**----- Start of picture text -----**<br>
1.8<br>1.6<br>1.4<br>1.2<br>1<br>0.8<br>0.6<br>0.4<br>0.2<br>0<br>0 100 200 300 400 500 600 700<br>Epoche<br>Train Loss Validation Loss<br>Loss →<br>**----- End of picture text -----**<br>


## _Abbildung 54: Trainings- und Validierungs-Loss des DiffusionNets_ 

Während der Trainings-Loss stetig kleiner wird, bleibt der Validierungs-Loss annähernd gleich und wird immer langsamer geringer. Die Divergenz zeigt hier, dass das Modell ab diesem Zeitpunkt schlechter auf ungesehene Daten generalisiert. Ein 

93 

6 Ergebnisse und Diskussion 

Abbruch des Trainings zum Verhindern von Overfitting ist hier nicht erforderlich. Dieser ist erst ab dem erneuten Anstieg des Validierungs-Loss notwendig und durch eine leicht parabelförmige Kurve des Validierungs-Loss gekennzeichnet. Da dies hier nicht vorliegt, ist ein Abbruch nicht nötig. Dies spiegelt sich ebenfalls im fast stetig gleichen Anstieg der Trainings- und Validierungsgenauigkeit wider, welche beide in Abbildung 55 dargestellt sind. Beide Werte nähern sich 80 % Genauigkeit an, wobei das Maximum des Trainings-Loss bei 80,47 % und des Validierungs-Loss bei 80,5 % liegt. Diese sehr nah beieinander liegenden Kennzahlen sprechen für eine gute Generalisierung. 

**==> picture [454 x 283] intentionally omitted <==**

**----- Start of picture text -----**<br>
0.9<br>0.8<br>0.7<br>0.6<br>0.5<br>0.4<br>0.3<br>0.2<br>0.1<br>0<br>0 100 200 300 400 500 600 700<br>Epoche<br>Train Genauigkeit Validation Genauigkeit<br>Genauigkeit →<br>**----- End of picture text -----**<br>


## _Abbildung 55: Trainings- und Validierungsgenauigkeit des DiffusionNets_ 

Entgegen der Genauigkeit, welche z.B. im Falle stark ungleichgewichteter Labels wie auch in diesem Datensatz nicht aussagekräftig sein kann, bietet sich der Dice-Score und Jaccard-Index besser an. Der Validierungs-Dice-Koeffizient ist in Abbildung 56 für jede Klasse sowie insgesamt dargestellt. Das Gehäuse wird mit Abstand am besten segmentiert was durch die Dominanz der Labels zu erklären ist. Diese wird zwar teilweise über den gewichteten NLL-Loss ausgeglichen, kann diesen Effekt jedoch aufgrund der Unkenntnis der Verteilung des Validierungsund Testdatensatzes nicht komplett ausgleichen. Die Kontaktierungen werden am zweitbesten segmentiert gefolgt von den Kabeleinführungen, Beschriftungsflächen und zuletzt den Aufrastpunkten. Diese Reihenfolge spiegelt genau die Verteilungen des Datensatzes wider. Deshalb ist damit zu rechnen, dass mit steigender Größe des Trainings-, Test- und Validierungsdatensatzes die Verteilung der realen Daten 

94 

6 Ergebnisse und Diskussion 

besser approximiert wird und sich die Metriken der segmentierten Klassen annähern. Die einzelnen Dice-Scores des Test-Datensatzes für die Klassen liegen bei 0,8506, 0,7144, 0,5219, 0,6161 und 0,5768. Daraus ergibt sich ein durchschnittlicher Validierungs-Dice-Score von 0,656. Die Jaccard-Indices auf dem Test-Datensatz liegen bei 0,7438, 0,5925, 0,3684, 0,4716 und 0,3943, woraus sich ein gesamter IoU von 0,5141 sowie ein gewichteter IoU von 0,7052 ergibt. 

**==> picture [454 x 398] intentionally omitted <==**

**----- Start of picture text -----**<br>
1<br>0.9<br>0.8<br>0.7<br>0.6<br>0.5<br>0.4<br>0.3<br>0.2<br>0.1<br>0<br>0 100 200 300 400 500 600 700<br>Epoche<br>Total Gehäuse Kontaktierung<br>Aufrastpunkt Kabeleinführung Beschriftungsfläche<br>Dice-Score →<br>**----- End of picture text -----**<br>


_Abbildung 56: Validierungs-Dice-Koeffizienten des DiffusionNets_ 

Zur Verbildlichung zeigt Abbildung 57 beispielhaft die segmentierte Komponente 0446017 sowie die realen Labels zu dieser. Auffällig ist, dass hier im Vergleich der Beschriftungsflächen zu viel segmentiert wurde, dies jedoch nur eine sehr geringe Auswirkung auf den Clusterschwerpunkt selbst hat. 

95 

6 Ergebnisse und Diskussion 

Cluster-Schwerpunkt 

Cluster-Schwerpunkt 

_Abbildung 57: Beispielhafte Segmentierung durch das trainierte DiffusionNet der Komponente 0446017_ 

96 

6 Ergebnisse und Diskussion 

Daraus lässt sich folgern, dass weniger die Genauigkeit der Segmentierung sondern die Identifikation der korrekten Anzahl an Instanzen herausfordernd ist. Aufgrund des geringen Einflusses einer leichten Abweichung der Segmentierung auf den ClusterSchwerpunkt, ist die Segmentierung in dem Maße erforderlich, dass die Cluster noch korrekt separiert werden können. 

## **MeshCNN** 

Im Gegensatz zum DiffusionNet wurde das Netzwerk auf Basis des MeshCNNs auf weniger Epochen trainiert, da die Berechnung um knapp eine Größenordnung aufwendiger ist und deshalb wesentlich mehr Zeit beansprucht. Der Grund hiefür liegt in der Verwendung der Mesh-Pooling und Mesh-Unpooling-Operation, welche weitgehend sequenziell statt parallel erfolgt. Aufgrund dieser Tatsache wird die GPU auch nur suboptimal genutzt und weist sehr viele Leistungsspitzen auf. Der Trainings- und Validierungs-Loss ist in Abbildung 58 dargestellt, wobei hier im Gegensatz zum DiffusionNet ein klar divergierender Trainings- und ValidierungsLoss vorliegt, da der Validierungs-Loss ab ca. Epoche 50 wieder ansteigt. Dies ist ein klares Indiz für Overfitting, weshalb die Ergebnisse des Modells stark an Aussagekraft verlieren. 

**==> picture [454 x 284] intentionally omitted <==**

**----- Start of picture text -----**<br>
1.8<br>1.6<br>1.4<br>1.2<br>1<br>0.8<br>0.6<br>0.4<br>0.2<br>0<br>0 50 100 150 200 250 300 350<br>Epoche<br>Train Loss Validation Loss<br>Loss →<br>**----- End of picture text -----**<br>


## _Abbildung 58: Trainings- und Validierungs-Loss des MeshCNN_ 

Trotz vieler Maßnahmen gegen Overfitting wie L2-Regularisierung, Dropout und geringere Modellgrößen, konnte das Overfitting in keinem Trainingslauf verhindert 

97 

6 Ergebnisse und Diskussion 

werden. Dies liegt insbesondere an der mangelnden Möglichkeit der HPO aufgrund langer Trainingszeiten. Dennoch wird die Performance des Netzwerks ebenfalls über den Dice-Koeffizienten und Jaccard-Index gemessen. Die Validierungs-Dice-Scores der einzelnen Klassen sowie der gesamte Dice-Score ist in Abbildung 59 dargestellt. Auffällig ist zum einen, dass die Ergebnisse wesentlich schlechter sind als die des DiffusionNets und zum anderen die einzelnen Klassen prozentual eine höhere Streuung im Dice-Koeffizienten aufweisen. Für die Segmentierung der Klassen Gehäuse, Kontaktierung und Aufrastpunkt hat dieses Netzwerk bessere Ergebnisse versprochen, weshalb der Ansatz weiterverfolgt wurde. Dieses Netzwerk empfiehlt sich jedoch aufgrund der wesentlich schlechteren Performance nicht. Eine genauere Untersuchung dieses Ansatzes, für den im Rahmen der Arbeit erstellten Datensatz, ist nur unter Verwendung größerer Batches sinnvoll, um herauszufinden, ob die Performance beim Training mehrerer Epochen wesentlich besser wird. Der hier dargestellte Trainingsdurchlauf wurde mit einer Batchgröße von zwei durchgeführt. Dementsprechend ist eine Vervierfachung des Grafikspeichers notwendig, sodass die Trainingszeit je Epoche in derselben Größenordnung des DiffusionNets liegt. Unter Berücksichtigung der Ausnutzung des Grafikspeichers, des Stromverbrauchs und der Tensoroperation pro Sekunde (TOPS) ist das MeshCNN allerdings ohnehin ungeeignet. 

**==> picture [454 x 312] intentionally omitted <==**

**----- Start of picture text -----**<br>
0.9<br>0.8<br>0.7<br>0.6<br>0.5<br>0.4<br>0.3<br>0.2<br>0.1<br>0<br>0 50 100 150 200 250 300 350<br>Epoche<br>Total Gehäuse Kontaktierung<br>Aufrastpunkt Kabeleinführung Beschriftungsfläche<br>Dice-Score →<br>**----- End of picture text -----**<br>


_Abbildung 59: Validierungs-Dice-Koeffizienten des MeshCNN_ 

98 

6 Ergebnisse und Diskussion 

## **YOLOv6** 

Anders als bei beiden vorherigen Ansätzen inferiert das implementierte YOLOv6Netzwerk auf 2D-Bildern der Größe 512×512 Pixel. Das Modell wurde nach [98] implementiert. Die verwendete Verlustfunktion untergliedert sich dabei in vier Bestandteile. Abbildung 60 stellt sowohl den gesamten Validierungs-Loss über alle Epochen dar sowie die einzelnen Bestandteile bestehend aus IoU-Loss (loss_iou), L1-Loss (loss_l1), Object-Loss (loss_obj) und CE-Loss (loss_cls). Der IoU-Loss repräsentiert dabei den Grad der Überlappung der tatsächlichen und inferierten Bounding-Boxen. Der L1-Loss misst wie nah die vorhergesagte Bounding-Box am Inferenzziel liegt. Die Messung und Bewertung, ob ein Objekt überhaupt in einem Rasterbereich des Bildes vorhanden ist oder nicht, erfolgt über den Object-Loss. Der CE-Loss optimiert die Korrektheit der inferierten Klasse einer Bounding-Box. Für alle vier Terme des gesamten Losses besteht das Ziel in der Minimierung und Annäherung an null. Der minimale Validierungs-Loss liegt in Epoche 365 vor und beträgt insgesamt 1,3697. Dieser setzt sich aus den Teilwerten 0,5731, 0,1824, 0,2802 sowie 0,3392 entsprechend der genannten Reihenfolge der Teilterme zusammen. 

**==> picture [454 x 287] intentionally omitted <==**

**----- Start of picture text -----**<br>
16<br>14<br>12<br>10<br>8<br>6<br>4<br>2<br>0<br>0 50 100 150 200 250 300 350<br>Epoche<br>Validation Loss loss_iou loss_l1 loss_obj loss_cls<br>Loss →<br>**----- End of picture text -----**<br>


_Abbildung 60: Validierungs-Loss des YOLOv6-CNN untergliedert in alle Bestandteile der Verlustfunktion_ 

Die Bewertung des letztlichen Ergebnisses erfolgt anhand der Mean Average Precision (mAP). Die Bemerkung mittels eines @-Zeichens meint hier den zur 

99 

6 Ergebnisse und Diskussion 

Berechnung verwendeten IoU-Schwellwert. Abbildung 61 stellt den Verlauf des Validierungs-mAP über 400 Trainingsepochen dar. Da Resultate über einem Schwellwert von 0,5 grundsätzlich als korrekt inferiert angesehen werden, wird hier der mAP@0,5 und zum anderen der mAP@[0,5:0,95] dargestellt. Letzterer ist der Durchschnitt aller mAPs von 0,5 bis 0,95 in 0,05 Intervallen und ist deshalb geringer, da für höhere mAPs mehr Gewissheit bei der Inferenz erforderlich ist. 

**==> picture [454 x 278] intentionally omitted <==**

**----- Start of picture text -----**<br>
1<br>0.9<br>0.8<br>0.7<br>0.6<br>0.5<br>0.4<br>0.3<br>0.2<br>0.1<br>0<br>0 50 100 150 200 250 300 350<br>Epoche<br>mAP@0,5 mAP@[0,5:0,95]<br>mAP →<br>**----- End of picture text -----**<br>


_Abbildung 61: Validierungs-mAP und mAP über IoU-Schwellwerte zwischen 0,5 und 0,95_ 

Der mAP@0,5 sowie mAP@[0,5:0,95] auf dem Testdatensatz liegt jeweils bei 0,9311 und 0,7279. Trotz dieser sehr guten Werte wird das YOLO-Netzwerk lediglich zur Inferenz der Aufrastpunkte in der finalen Pipeline verwendet. Der Grund hierfür liegt darin, dass nicht alle Klassen immer aus den sechs betrachteten Positionen erkannt werden können. Kontaktierungen, die schräg in das Bauteil führen und teilweise von oben oder der Seite verdeckt sind, stellen hierfür ein beispielhaftes Problem dar. Abbildung 62 stellt exemplarisch ein solches Problem an der Komponente 1005331 dar. Dieses Beispiel zeigt in Bauteilansicht (b) von oben, dass die Kontaktierungen nicht ganz, jedoch teilweise verdeckt sind. In einem solchen Fall können verschiedene Komplikationen auftreten. Zum einen wird das Nichterkennen der Kontaktierung aufgrund des Winkels wesentlich wahrscheinlicher. Zum anderen kann es vorkommen, dass die gleichen Kontaktierungen in verschiedenen Ansichten detektiert werden und eine anschließende Zuordnung erforderlich ist. Diese Zuordnung wird insbesondere dann enorm komplex, sobald die Anzahl der 

100 

6 Ergebnisse und Diskussion 

detektierten Kontaktierungen in (b) und (c) unterschiedlich ist. Ebenfalls besteht die Möglichkeit eines komplett verdeckten Anschlusses aus allen sechs Perspektiven. Aufgrund der aufgeführten Probleme empfiehlt sich das Netzwerk, trotz des sehr guten mAP, nicht alleinig zur Identifikation der einzelnen Klassen. Stattdessen rechtfertigt es einen hybriden Lösungsansatz zur Kombination eines intrinsischen, direkt auf den Daten operierenden Modells und einer abstrakteren Lösung. Die Verwendung des YOLO-Netzwerks in Kombination mit dem DiffusionNet bietet demnach den besten Lösungsansatz. Die Verwendung eines solchen YOLO-CNNs für die Klassen Kontaktierung und Kabeleinführung ist nicht empfehlenswert, da das Problem nicht über mehr gelabelte Daten gelöst werden kann und Fälle auftreten können, die unter keinen Umständen technisch erkannt werden können. Dieses Problem legt ebenfalls nahe ein 3D-basiertes YOLO-Netzwerk zu erproben. Hierdurch könnten durch das Agieren auf Punktwolken, die eben genannten Komplikationen teilweise umgangen werden. Ein teilautomatisiertes Erstellen der 3DBounding-Box Labels ausgehend von den Scheitelpunkt-Labels ist ebenso denkbar. 

**==> picture [296 x 295] intentionally omitted <==**

**----- Start of picture text -----**<br>
(c)<br>I<br>(a)<br>(a) vorne (b) oben (c) hinten<br>**----- End of picture text -----**<br>


_Abbildung 62: Exemplarische Darstellung eines problematischen Bauteils für die YOLO-Inferenz am Beispiel der Komponente 1005331_ 

Die Inferenzpipeline verwendet deshalb schlussendlich das YOLO-Netzwerk ausschließlich zur Detektion der Aufrastpunkte, da bei diesen die Sichtbarkeit von mindestens zwei Ansichten immer gegeben ist. Anderenfalls ist das Klemmen des 

101 

6 Ergebnisse und Diskussion 

Bauteils auf eine Schiene nicht möglich. Nachfolgend wird zudem durch Abbildung 63 deutlich, dass die Detektion der Aufrastpunkte durch das YOLO-Netzwerk bei einem weiteren Problem hilft. In (a) ist hierbei ersichtlich, wo die zwei Schienen verlaufen, und die Komponente festgeklemmt wird. In (b) wird dabei deutlich, dass die Aufrastpunkte der rechten Schiene in (a) in zwei Teile untergliedert ist. Bei einer Segmentierung durch das DiffusionNet ist in solchen Fällen die Zuordnung aller segmentierten Scheitelpunkte der Klasse „Aufrastpunkt” zur jeweils korrekten Schiene sehr schwierig, weshalb der Ansatz mittels des YOLO-CNNs hier ebenfalls hilfreich ist. 

**==> picture [278 x 11] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a) rechts (b) hinten<br>**----- End of picture text -----**<br>


_Abbildung 63: Detektion der Aufrastpunkte durch das YOLO-CNN_ 

Da die gelabelten Bilder aus FreeCAD stammen, die Qualität der durch OCCT gerenderten Bilder jedoch wesentlich besser ist, muss das YOLO-CNN ebenfalls auf den OCCT-Bildern gute Vorhersagen erzeugen können. Obwohl trotz der fehlenden Labels die quantitative Analyse für diese Bilder nicht möglich ist, wird bei der Inferenz und manuellen Beurteilung der Vorhersagen auf den OCCT-Bildern deutlich, dass das neuronale Netz quasi identisch gut performt. In Abbildung 64 ist links die Inferenz auf Bildern zu sehen, welche von FreeCAD generiert wurden. Auf der rechten Seite sind die Vorhersagen der zugehörigen identischen Bilder, welche von OCCT generiert wurden. Bei genauer Betrachtung fällt auf, dass zwar minimale Unterschiede in der Größe der Bounding-Boxen bestehen, jedoch diese grundsätzlich identisch vorhergesagt werden. Dennoch ist das Labeln für weitere Arbeiten basierend auf den OCCT-Bildern empfehlenswert, um die Verteilungen der Testdaten schon während des Trainings besser zu repräsentieren. Darüber hinaus können mittels der OCCT-Bilder sogar leicht bessere Ergebnisse aufgrund der höheren Bildqualität erwartet werden. Insbesondere die genaueren Farbgradienten, welche Schrägen im Bauteil darstellen, beinhalten hier zusätzliche Informationen, welche den mAP verbessern können. Zusätzlich sind aufgrund der höheren Ausfüllung des Bildes bei identischer Größe die Features auch entsprechend größer dargestellt. 

102 

6 Ergebnisse und Diskussion 

_Abbildung 64: Vergleich der Vorhersagen auf von FreeCAD generierten Bildern (links) sowie auf von OCCT generierten Bildern (rechts)_ 

103 

6 Ergebnisse und Diskussion 

Da größere Bounding-Boxen meist leichter oder schneller zu erlernen sind, spricht auch dies für noch bessere Metriken beim Trainieren auf gelabelten OCCT-Bildern. 

## **6.2.3 Nutzen und Limitationen der neuronalen Netze** 

Ausgehend von den Vor- und Nachteilen der modellierten neuronalen Netze, deren Komplikationen, Limitationen sowie den schließlich erzielten Ergebnissen empfiehlt sich eine Gegenüberstellung dieser. Das modellierte MeshCNN kann zwar ebenfalls die Dreiecksnetze der Komponenten segmentieren benötigt hierfür jedoch mehr als die 20-fache Dauer. Dadurch können die Hyperparameter wesentlich schlechter optimiert werden, da die vielen einzelnen Trainingsläufe ebenfalls zeitaufwendiger sind. Der Output liegt als Tensor mit 18000 Einträgen vor, welche die Kantenmerkmale repräsentieren. Aufgrund der nachträglichen Kalkulation der Metriken, welche teilweise auf die Scheitelpunkte angewiesen sind, um Cluster-, Volumen- und Flächenschwerpunkte zu berechnen, ist ein Mapping der Kantenmerkmale auf Scheitelpunktmerkmale erforderlich. Zusätzlich liegt trotz ähnlicher Segmentierungswerte, eine wesentlich höhere Varianz innerhalb der fünf Klassen vor. Die gewichtete Verlustfunktion hatte hier keinen ausreichend wirksamen Effekt für den Ausgleich der stark unterschiedlichen Häufigkeiten der Labels. Zudem liegt eine stärkere Tendenz zum Overfitting vor, was durch einen Anstieg des Validierungs-Loss signalisiert wird. Dies bestätigt ebenfalls die in [56] dargelegte Behauptung einer hohen Tendenz zum Overfitting des MeshCNN auf die MeshStruktur ohne die wichtigen geometrischen Features der Mannigfaltigkeit zu erlernen. Besonders beim Verwenden verschiedener Mesh-Algorithmen und Netzunterteilungen spielt dies eine entscheidende Rolle. Gegensätzlich hierzu konnte das DiffusionNet sowohl in wesentlich kürzerer Zeit als auch geringerer Varianz innerhalb der Segmentierungsklassen trainiert werden. Der Test-IoU, welcher leicht über dem Validierungs-IoU liegt, spricht für sehr gute Generalisierung. Eine vollständige Eliminierung der Dominanz der Gehäuselabels konnte auch hier nicht komplett entgegengewirkt werden. Der Effekt ist jedoch geringer und alle Metriken je Klasse liegen näher aneinander, sodass keine Klasse sehr schlecht segmentiert wird. Anders als MeshCNN inferiert das DiffusionNet direkt die Scheitelpunkte, weshalb keine nachträgliche Konvertierung der Vorhersagen nötig ist. Laut [56] ist ebenfalls das Training und die Inferenz über Dreiecksflächen möglich. Ebenfalls ist beim DiffusionNet die Anpassung möglich, um Punktwolken verarbeiten zu können. [56] Auch wenn im Rahmen dieser Arbeit beide dieser Möglichkeiten nicht genutzt wurden, bietet das DiffusionNet eine flexiblere Modellarchitektur. Vor allem aufgrund letzterer Option zur Inferenz von Punktwolken ergibt sich die Möglichkeit reale Komponenten über Tiefenkameras zu scannen und auf Basis der generierten Punktwolken geometrische Merkmale zu erkennen. Ein Nachteil des DiffusionNets liegt in der Kalkulation der Laplace-, Massen-, und 

104 

6 Ergebnisse und Diskussion 

Gradientenmatrix sowie mehrerer Eigenvektoren. Im Trainingslauf ist dies vernachlässigbar, da die Matrizen nur einmal kalkuliert und dann in einem Cache zwischengespeichert werden. Während der Inferenz auf realen Daten ist eine Neuberechnung dieser Matrizen jedoch für jede Komponente erforderlich. Dies führt zu einer ungefähren Inferenzzeit des modellierten DiffusionNets von ca. 1 Sekunde pro Bauteil. Das MeshCNN weist dieses Problem nicht auf, ist dagegen jedoch sehr speicherineffizient und erzeugt größere Schwankungen in der GPU-Auslastung aufgrund zwischenzeitlich benötigter sehr großer Matrizen. Daher besteht bezüglich der Inferenzzeit keine Tendenz für eines der beiden Netzwerke. 

Da beide Modelle die Aufrastpunkte der Komponenten am schlechtesten segmentieren, liegt nahe eine weitere Inferenzmöglichkeit einzubinden. Das YOLOCNN wird deshalb ausschließlich für die Aufrastpunkte verwendet, da nur bei dieser Klasse die Sichtbarkeit der Merkmale von zwei der sechs Ansichten garantiert ist. Ergänzend ist bei allen Aufrastpunkten das Merkmal immer über die gesamte Breite der Komponente vorhanden, weshalb es sich nicht nur leicht aus einer zweidimensionalen Perspektive detektieren lässt, sondern die vorhergesagte Bounding-Box auch direkt in drei Dimensionen, unter Kenntnis der Perspektive, überführt werden kann. Die Kombination eines geometrischen sowie eines herkömmlichen CNN bietet sich deshalb an. Aufgrund der genannten Vorteile des DiffusionNets empfiehlt sich die Verwendung dieses Modells gegenüber eines MeshCNN nach [60] oder [61]. 

105 

7 Zusammenfassung und Ausblick 

## **7 Zusammenfassung und Ausblick** 

Das Ziel dieser Arbeit bestand in der Implementierung und Evaluation geeigneter Verfahren zur automatischen Identifikation und Positionsbestimmung geometrischer Features von Reihenklemmen ausgehend von STEP-Dateien. Die Schaltschrankkomponenten weisen vier grundlegende geometrische Features auf. Dementsprechend wurden verschiedene Ansätze zur semantischen Segmentierung von Dreiecksoberflächennetzen verfolgt und implementiert. Die Segmentierung erfolgte stets in die Klassen Gehäuse, Kontaktierung, Aufrastpunkte, Kabeleinführung und Beschriftungsflächen. Die Vorgehensweise orientierte sich an der DMME-Methodik. Zusätzlich wurde ein gesamter Datensatz aus dem Internet via Web-Scraping aggregiert. Dieser umfasst ca. 200 GB an STEP-Dateien. Im Rahmen der Datenvorverarbeitung wurden geometrische Duplikate entfernt, Filter auf Basis der Bauteilmetadaten angewandt, die Modelle einheitlich skaliert und die CADModelle in verschiedene Dateiformate konvertiert. Es wurden zwei verschiedene DLModelle erprobt. Das DiffusionNet war dabei mit Blick auf die Rechenkapazität nicht nur das effizienteste, sondern auch das beste Modell unter Evaluierung des JaccardIndex und Dice-Koeffizienten. Neben diesem Modell ist das MeshCNN zwar in der Lage Features zu erlernen, nicht jedoch in dem Umfang und in der Trainingszeit des DiffusionNets. Es wurde über alle fünf Segmentierungsklassen ein Dice-Score von 65,6 % sowie ein Jaccard-Index von 51,41% erzielt. Der IoU unterteilt sich auf die einzelnen Klassen wie folgt: 74,38 %, 59,25 %, 36,84 %, 47,16 % und 39,43 %. Aufgrund eines starken Ungleichgewichts der Labels empfiehlt sich zudem das Heranziehen des gewichteten Jaccard-Index, welcher 70,52 % beträgt. In der Datenaugmentierungspipeline wurden zufällige Rotationen, Rauschen und Deformationen mittels des As-Rigid-As-Possible-Algorithmus implementiert. Da die evaluierten Clustering-Algorithmen nur bedingt funktioniert haben, wurde die Segmentierung einer Klasse in einzelne Instanzen durch Algorithmen der Graphentheorie gelöst. Das Hochskalieren der Daten auf mehr Knoten, Kanten und Dreiecke bewirkte weder bessere noch schlechtere Ergebnisse, ist jedoch für weitere Arbeiten dennoch zur Verbesserung der graphenbasierten Instanz-Segmentierung zu empfehlen. Hier ist eine positive Korrelation zwischen der Anzahl der Knoten und der korrekt segmentierten Instanzen erkennbar. Der Jaccard-Index von über 50 % legt außerdem nahe, den gesamten Ansatz weiter zu verfolgen. Mögliche zukünftige Vorgehensweisen sind beispielsweise das Labeln neuer Daten, oder die Rückführung der Modellvorhersagen in das Labelingtool, um eine Feedbackschleife zu erstellen. Letzteres ist insbesondere aufgrund des sehr zeitaufwändigen LabelingProzesses der Dreiecksnetze empfehlenswert. Ein alternativer Ansatz besteht in einer hybriden Vorgehensweise, um nicht nur auf Basis von Dreiecksnetzen, sondern auch Metadaten zu trainieren. Die aggregierten Links zu den Bauteildokumentationen der Hersteller bieten zudem die Möglichkeit eine Vielzahl 

106 

7 Zusammenfassung und Ausblick 

weiterer Daten zu scrapen und automatisiert auszulesen. Die Ergebnisse zeigen auf, dass das Erkennen von Merkmalen hochvariabler Bauteile ausschließlich basierend auf der Geometrie möglich ist. Auffällig ist zudem, dass insbesondere Bauteile besser segmentiert werden, deren Features in ähnlicher Form im gleichen Datensatz vorhanden sind. Dies bietet eine argumentative Grundlage neue Daten zu labeln. Für ein gesamtheitliches Konzept belaufen sich die nächsten Schritte auf das Erkennen von gesamten Bauteilkomponenten in einem Schaltschrank. Die einzeln erkannten Schaltschrankkomponenten, könnten dann mittels der in dieser Arbeit beschriebenen Wege teilsegmentiert werden. Die Ergebnisse verdeutlichen abschließend, dass der strukturierte Informationsgewinn aus dem erarbeiteten Datensatz definitiv möglich ist, jedoch aktuell noch auf eine ergänzende Hilfe bei der Analyse von hochvariablen Baugruppen beschränkt ist. Für einen autonomen Einsatz in der Produktion sollte das Modell zudem in allen vorzuweisenden Kategorien mindestens einen JaccardIndex von 50 % aufweisen sowie in eine Software-Umgebung nach den Prinzipien der Machine Learning Operations integriert sein, um die stetige Verbesserung des Modells zu ermöglichen, Datendrifts zu verhindern und Metriken während der Inferenz bereitzustellen. 

107 

Literaturverzeichnis 

## **Literaturverzeichnis** 

- [1] TEMPEL, P., F. EGER, A. LECHLER und A. VERL. _Schaltschrankbau 4.0. Eine Studie über die Automatisierungs- und Digitalisierungspotentiale in der Fertigung von Schaltschränken und Schaltanlagen im klassischen Maschinen- und Anlagenbau,_ 2017 

- [2] HEFNER, F., S. SCHMIDBAUER und J. FRANKE. Pose error correction of a robot end-effector using a 3D visual sensor for control cabinet wiring [online]. _Procedia CIRP,_ 2020, **93** , S. 1133-1138. ISSN 22128271. Verfügbar unter: doi:10.1016/j.procir.2020.04.088 

- [3] HEFNER, F., S. SCHMIDBAUER und J. FRANKE. Vision-based adjusting of a digital model to real-world conditions for wire insertion tasks [online]. _Procedia CIRP,_ 2021, **97** , S. 342-347. ISSN 22128271. Verfügbar unter: doi:10.1016/j.procir.2020.05.248 

- [4] MORAVEC, H. When will computer hardware match the human brain? _Journal of Evolution and Technology,_ 1998, **1** 

- [5] CIRILLO, P., G. LAUDANTE und S. PIROZZI. Vision-Based Robotic Solution for Wire Insertion With an Assigned Label Orientation [online]. _IEEE Access,_ 2021, **9** , S. 102278-102289. Verfügbar unter: 

   - doi:10.1109/ACCESS.2021.3098472 

- [6] ZHANG, Y., W. LIANG, M. YUAN, J. XIAO, J. LI und S. PENG. Real-time State Recognition of Switches on Electrical Cabinet Panel Using Hybrid Visual Features. In: _2020 IEEE 18th International Conference on Industrial Informatics (INDIN):_ IEEE, 20. Juli 2020 - 23. Juli 2020, S. 920-925. ISBN 9781-7281-4964-6 

- [7] SPIES, S., M. BARTELT und B. KUHLENKOTTER. Wiring of Control Cabinets using a Distributed Control within a Robot-Based Production Cell. In: _2019 19th International Conference on Advanced Robotics (ICAR):_ IEEE, 2. Dezember 2019 - 6. Dezember 2019, S. 332-337. ISBN 978-1-7281-24674 

- [8] IAN GOODFELLOW, YOSHUA BENGIO und AARON COURVILLE. _Deep Learning:_ MIT Press, 2016 

- [9] SAMUEL, A.L. Some Studies in Machine Learning Using the Game of Checkers [online]. _IBM Journal of Research and Development,_ 1959, **3** (3), S. 210-229. ISSN 0018-8646. Verfügbar unter: doi:10.1147/rd.33.0210 

- [10] CHEN, M.-Y., H.-S. CHIANG, E. LUGHOFER und E. EGRIOGLU. Deep learning: emerging trends, applications and research challenges [online]. _Soft_ 

108 

Literaturverzeichnis 

_Computing,_ 2020, **24** (11), S. 7835-7838. ISSN 1432-7643. Verfügbar unter: doi:10.1007/s00500-020-04939-z 

- [11] VDMA SOFTWARE UND DIGITALISIERUNG. _Quick Guide: Machine learning im Maschinen- und Anlagenbau._ Frankfurt am Main, 2018 

- [12] KELLEHER, J.D. _Deep learning._ Cambridge, Massachusetts: The MIT Press, 2019. The MIT press essential knowledge series. ISBN 0262537559 

- [13] DEISENROTH, M.P., A.A. FAISAL und C.S. ONG. _Mathematics for machine learning._ Cambridge: Cambridge University Press, 2020. ISBN 110845514X 

- [14] SUTTON, R.S. und A. BARTO. _Reinforcement learning. An introduction._ Second edition. Cambridge, Massachusetts: The MIT Press, 2018. Adaptive computation and machine learning. ISBN 0262039249 

- [15] SHALEV-SHWARTZ, S. und S. BEN-DAVID. _Understanding machine learning. From theory to algorithms._ Cambridge: Cambrige University Press, 2014. ISBN 1107057132 

- [16] GOYAL, P., M. CARON, B. LEFAUDEUX, M. XU, P. WANG, V. PAI, M. SINGH, V. LIPTCHINSKY, I. MISRA, A. JOULIN und P. BOJANOWSKI. _Selfsupervised Pretraining of Visual Features in the Wild:_ arXiv, 2021 

- [17] BARDES, A., J. PONCE und Y. LECUN. _VICRegL: Self-Supervised Learning of Local Visual Features:_ arXiv, 2022 

- [18] FROCHTE, J. _Maschinelles Lernen. Grundlagen und Algorithmen in Python._ 3., überarbeitete und erweiterte Auflage. München: Hanser, 2021. Plus.Hanser-Fachbuch. ISBN 978-3-446-46144-4 

- [19] MICHELUCCI, U. _Applied Deep Learning. A Case-Based Approach to Understanding Deep Neural Networks._ Berkeley, CA: Apress L. P, 2018. ISBN 9781484237908 

- [20] AYYADEVARA, V.K. und Y. REDDY. _Modern computer vision with PyTorch. Explore deep learning concepts and implement over 50 real-world image applications._ Place of publication not identified: Packt Publishing, 2020. ISBN 1839213477 

- [21] RUDER, S. _An overview of gradient descent optimization algorithms:_ arXiv, 2016 

- [22] DOGO, E.M., O.J. AFOLABI, N.I. NWULU, B. TWALA und C.O. AIGBAVBOA. A Comparative Analysis of Gradient Descent-Based Optimization Algorithms on Convolutional Neural Networks. In: _2018 International Conference on Computational Techniques, Electronics and Mechanical Systems (CTEMS):_ IEEE, 21. Dezember 2018 - 22. Dezember 2018, S. 92-99. ISBN 978-1-5386-7709-4 

109 

Literaturverzeichnis 

- [23] DAUPHIN, Y., R. PASCANU, C. GULCEHRE, K. CHO, S. GANGULI und Y. BENGIO. _Identifying and attacking the saddle point problem in highdimensional non-convex optimization:_ arXiv, 2014 

- [24] DUCHI, J., E. HAZAN und Y. SINGER. Adaptive Subgradient Methods for Online Learning and Stochastic Optimization. _Journal of Machine Learning Research,_ 2011, **12** , S. 2121-2159 

- [25] ZEILER, M.D. _ADADELTA: An Adaptive Learning Rate Method:_ arXiv, 2012 [26] LECUN, Y., L. BOTTOU, G. ORR und K.-R. MÜLLER. Efficient BackProp, 2000 

- [27] KINGMA, D.P. und J. BA. _Adam: A Method for Stochastic Optimization:_ arXiv, 2014 

- [28] GLOROT, X., A. BORDES und Y. BENGIO. Deep Sparse Rectifier Neural Networks. In: G. GORDON, D. DUNSON und M. DUDÍK, Hg. _Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics._ Fort Lauderdale, FL, USA: PMLR, 2011, S. 315-323 

- [29] PEDAMONTI, D. _Comparison of non-linear activation functions for deep neural networks on MNIST classification task:_ arXiv, 2018 

- [30] REN, S., K. HE, R. GIRSHICK und J. SUN. _Faster R-CNN: Towards RealTime Object Detection with Region Proposal Networks:_ arXiv, 2015 

- [31] GIRSHICK, R. _Fast R-CNN:_ arXiv, 2015 [32] HE, K., G. GKIOXARI, P. DOLLÁR und R. GIRSHICK. _Mask R-CNN:_ arXiv, 2017 

- [33] PASZKE, A., S. GROSS, F. MASSA, A. LERER, J. BRADBURY, G. CHANAN, T. KILLEEN, Z. LIN, N. GIMELSHEIN, L. ANTIGA, A. DESMAISON, A. KOPF, E. YANG, Z. DEVITO, M. RAISON, A. TEJANI, S. CHILAMKURTHY, B. STEINER, L. FANG, J. BAI und S. CHINTALA. PyTorch: An Imperative Style, High-Performance Deep Learning Library. In: _Advances in Neural Information Processing Systems 32:_ Curran Associates, Inc, 2019, S. 8024-8035 

- [34] WANG, Q., Y. MA, K. ZHAO und Y. TIAN. A Comprehensive Survey of Loss Functions in Machine Learning [online]. _Annals of Data Science,_ 2022, **9** (2), S. 187-212. ISSN 2198-5812. Verfügbar unter: doi:10.1007/s40745-02000253-5 

- [35] LIAW, R., E. LIANG, R. NISHIHARA, P. MORITZ, J.E. GONZALEZ und I. STOICA. _Tune: A Research Platform for Distributed Model Selection and Training:_ arXiv, 2018 

110 

Literaturverzeichnis 

- [36] FEURER, M. und F. HUTTER. Hyperparameter Optimization. In: F. HUTTER, L. KOTTHOFF und J. VANSCHOREN, Hg. _Automated Machine Learning._ Cham: Springer International Publishing, 2019, S. 3-33. ISBN 9783-030-05317-8 

- [37] BERGSTRA, J., R. BARDENET, Y. BENGIO und B. KÉGL. Algorithms for Hyper-Parameter Optimization. In: J. SHAWE-TAYLOR, R. ZEMEL, P. BARTLETT, F. PEREIRA und K.Q. WEINBERGER, Hg. _Advances in Neural Information Processing Systems:_ Curran Associates, Inc, 2011 

- [38] BERGSTRA, J. und Y. BENGIO. Random Search for Hyper-Parameter Optimization. _The Journal of Machine Learning Research,_ 2012, **13** , S. 281305 

- [39] HORNIK, K., M. STINCHCOMBE und H. WHITE. Multilayer feedforward networks are universal approximators [online]. _Neural Networks,_ 1989, **2** (5), S. 359-366. ISSN 08936080. Verfügbar unter: doi:10.1016/08936080(89)90020-8 

- [40] KRIZHEVSKY, A., I. SUTSKEVER und G.E. HINTON. ImageNet Classification with Deep Convolutional Neural Networks. In: F. PEREIRA, C.J. BURGES, L. BOTTOU und K.Q. WEINBERGER, Hg. _Advances in Neural Information Processing Systems:_ Curran Associates, Inc, 2012 

- [41] BYUN, A. _Convolutional Neural Networks for Visual Recognition._ [online] [Zugriff am: 26. Oktober 2022]. Verfügbar unter: http://cs231n.github.io/neural-networks-1 

- [42] LECUN, Y., B. BOSER, J.S. DENKER, D. HENDERSON, R.E. HOWARD, W. HUBBARD und L.D. JACKEL. Backpropagation Applied to Handwritten Zip Code Recognition [online]. _Neural Computation,_ 1989, **1** (4), S. 541-551. ISSN 0899-7667. Verfügbar unter: doi:10.1162/neco.1989.1.4.541 

- [43] SCHWARTZ, D.B., V.K. SAMALAM, S.A. SOLLA und J.S. DENKER. Exhaustive Learning [online]. _Neural Computation,_ 1990, **2** (3), S. 374-385. ISSN 0899-7667. Verfügbar unter: doi:10.1162/neco.1990.2.3.374 

- [44] JIANG, X., A. HADID, Y. PANG, E. GRANGER und X. FENG. _Deep Learning in Object Detection and Recognition._ Singapore: Springer Singapore, 2019. ISBN 978-981-10-5151-7 

- [45] ZHUANG, F., Z. QI, K. DUAN, D. XI, Y. ZHU, H. ZHU, H. XIONG und Q. HE. _A Comprehensive Survey on Transfer Learning:_ arXiv, 2019 

- [46] RUSSAKOVSKY, O., J. DENG, H. SU, J. KRAUSE, S. SATHEESH, S. MA, Z. HUANG, A. KARPATHY, A. KHOSLA, M. BERNSTEIN, A.C. BERG und L. FEI-FEI. _ImageNet Large Scale Visual Recognition Challenge:_ arXiv, 2014 

111 

Literaturverzeichnis 

- [47] BURNHAM, K.P. und D.R. ANDERSON. _Model selection and multimodel inference. A practical information-theoretic approach._ 2. ed. New York, NY: Springer, 2010. ISBN 1441929738 

- [48] TUMER, K. und J. GHOSH. Estimating the Bayes error rate through classifier combining. In: _Proceedings of 13th International Conference on Pattern Recognition:_ IEEE, 29. August 1996 - 29. August 1996, 695-699 vol.2. ISBN 0-8186-7282-X 

- [49] THOMAS, G. _Mathematics for Machine Learning_ [online], 2018 [Zugriff am: 26. Oktober 2022]. Verfügbar unter: https://gwthomas.github.io/docs/math4ml.pdf 

- [50] CLARK, K. _Computing Neural Network Gradients_ [online] [Zugriff am: 27. Oktober 2022]. Verfügbar unter: 

   - https://web.stanford.edu/class/cs224n/readings/gradient-notes.pdf 

- [51] BRONSTEIN, M.M., J. BRUNA, Y. LECUN, A. SZLAM und P. VANDERGHEYNST. Geometric deep learning: going beyond Euclidean data [online], 2016. Verfügbar unter: doi:10.48550/arXiv.1611.08097 

- [52] BRONSTEIN, M.M., J. BRUNA, T. COHEN und P. VELIČKOVIĆ. _Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges:_ arXiv, 2021 

- [53] FLEGG, H.G. _From geometry to topology._ Mineola, NY: Dover Publ, 2001. ISBN 0486419614 

- [54] LEE, J.M. _Introduction to topological manifolds._ Second edition. New York: Springer, 2011. Graduate texts in mathematics. 202. ISBN 1441979395 

- [55] WANG, Y., Y. SUN, Z. LIU, S.E. SARMA, M.M. BRONSTEIN und J.M. SOLOMON. Dynamic Graph CNN for Learning on Point Clouds. _ACM Transactions on Graphics (TOG),_ 2019 

- [56] SHARP, N., S. ATTAIKI, K. CRANE und M. OVSJANIKOV. _DiffusionNet: Discretization Agnostic Learning on Surfaces:_ arXiv, 2020 

- [57] POULENARD, A., M.-J. RAKOTOSAONA, Y. PONTY und M. OVSJANIKOV. _Effective Rotation-invariant Point CNN with Spherical Harmonics kernels:_ arXiv, 2019 

- [58] CHANG, A.X., T. FUNKHOUSER, L. GUIBAS, P. HANRAHAN, Q. HUANG, Z. LI, S. SAVARESE, M. SAVVA, S. SONG, H. SU, J. XIAO, L. YI und F. YU. _ShapeNet: An Information-Rich 3D Model Repository,_ 2015 

- [59] LIAN, Z., A. GODIL, B. BUSTOS, M. DAOUDI, J. HERMANS, S. KAWAMURA, Y. KURITA, G. LAVOUÉ, H.V. NGUYEN, R. OHBUCHI, Y. OHKITA, Y. OHISHI, F. PORIKLI, M. REUTER, I. SIPIRAN, D. SMEETS, P. SUETENS, H. TABIA und D. VANDERMEULEN. _SHREC '11 Track: Shape_ 

112 

Literaturverzeichnis 

   - _Retrieval on Non-rigid 3D Watertight Meshes:_ The Eurographics Association, 2011 

- [60] SCHNEIDER, L., A. NIEMANN, O. BEUING, B. PREIM und S. SAALFELD. _MedMeshCNN -- Enabling MeshCNN for Medical Surface Models:_ arXiv, 2020 

- [61] HANOCKA, R., A. HERTZ, N. FISH, R. GIRYES, S. FLEISHMAN und D. COHEN-OR. MeshCNN: A Network with an Edge [online], 2018. Verfügbar unter: doi:10.48550/arXiv.1809.05910 

- [62] HUBER, S., H. WIEMER, D. SCHNEIDER und S. IHLENFELDT. DMME: Data mining methodology for engineering applications – a holistic extension to the CRISP-DM model [online]. _Procedia CIRP,_ 2019, **79** , S. 403-408. ISSN 22128271. Verfügbar unter: doi:10.1016/j.procir.2019.02.106 

- [63] WIEMER, H., L. DROWATZKY und S. IHLENFELDT. Data Mining Methodology for Engineering Applications (DMME)—A Holistic Extension to the CRISP-DM Model [online]. _Applied Sciences,_ 2019, **9** (12), S. 2407. Verfügbar unter: doi:10.3390/app9122407 

- [64] SCHAFER, F., C. ZEISELMAIR, J. BECKER und H. OTTEN. Synthesizing CRISP-DM and Quality Management: A Data Mining Approach for Production Processes. In: _2018 IEEE International Conference on Technology Management, Operations and Decisions (ICTMOD):_ IEEE, 21. November 2018 - 23. November 2018, S. 190-195. ISBN 978-1-5386-4315-0 

- [65] PETER CHAPMAN, JANET CLINTON, RANDY KERBER, TOM KHABAZA, THOMAS P. REINARTZ, COLIN SHEARER und RICHARD WIRTH. CRISPDM 1.0: Step-by-step data mining guide. In: , 2000 

- [66] SHEARER, C. The CRISP-DM Model: The New Blueprint for Data Mining. _Journal of Data Warehousing,_ 2000, **5** (4) 

- [67] LOURIDAS, P. Static code analysis [online]. _IEEE Software,_ 2006, **23** (4), S. 58-61. ISSN 0740-7459. Verfügbar unter: doi:10.1109/MS.2006.114 

- [68] _Python Enhancement Proposal (PEP)_ [online] [Zugriff am: 27. Oktober 2022]. Verfügbar unter: https://peps.python.org/pep-0000/ 

- [69] HATTORI, H. _Autopep8_ [online] [Zugriff am: 27. Oktober 2022]. Verfügbar unter: https://github.com/hhatto/autopep8 

- [70] VAN OORT, B., L. CRUZ, M. ANICHE und A. VAN DEURSEN. The Prevalence of Code Smells in Machine Learning projects. In: _2021 IEEE/ACM 1st Workshop on AI Engineering - Software Engineering for AI (WAIN):_ IEEE, 30. Mai 2021 - 31. Mai 2021, S. 1-8. ISBN 978-1-6654-4470-5 

113 

Literaturverzeichnis 

- [71] PYTHON SOFTWARE FOUNDATION. _Python Documentation_ [online] [Zugriff am: 27. Oktober 2022]. Verfügbar unter: https://docs.python.org/3/ 

- [72] _Sphinx Python Documentation Generator_ [online] [Zugriff am: 27. Oktober 2022]. Verfügbar unter: https://www.sphinxdoc.org/en/master/usage/quickstart.html 

- [73] CONDA. _Conda Docs_ [online] [Zugriff am: 27. Oktober 2022]. Verfügbar unter: https://docs.conda.io/projects/conda/en/stable/ 

- [74] GUIDO VAN ROSSUM. _The Python/C API_ [online], 2022 [Zugriff am: 27. Oktober 2022]. Verfügbar unter: https://docs.python.org/3/c-api/index.html 

- [75] OLIPHANT, T.E. _Guide to NumPy._ 2nd edition. Austin, Texas: Continuum Press, September 2015. ISBN 151730007X 

- [76] HARRIS, C.R., K.J. MILLMAN, S.J. VAN DER WALT, R. GOMMERS, P. VIRTANEN, D. COURNAPEAU, E. WIESER, J. TAYLOR, S. BERG, N.J. SMITH, R. KERN, M. PICUS, S. HOYER, M.H. VAN KERKWIJK, M. BRETT, A. HALDANE, J. DEL FERNÁNDEZ RÍO, M. WIEBE, P. PETERSON, P. GÉRARD-MARCHANT, K. SHEPPARD, T. REDDY, W. WECKESSER, H. ABBASI, C. GOHLKE und T.E. OLIPHANT. Array programming with NumPy [online]. _Nature,_ 2020, **585** , S. 357-362. Verfügbar unter: doi:10.1038/s41586-020-2649-2 

- [77] GHORPADE, J., J. PARANDE, M. KULKARNI und A. BAWASKAR. GPGPU Processing in CUDA Architecture [online], 2012. Verfügbar unter: doi:10.48550/arXiv.1202.4347 

- [78] DEHAL, R.S., C. MUNJAL, A.A. ANSARI und A.S. KUSHWAHA. GPU Computing Revolution: CUDA. In: _2018 International Conference on Advances in Computing, Communication Control and Networking (ICACCCN):_ IEEE, 12. Oktober 2018 - 13. Oktober 2018, S. 197-201. ISBN 978-1-5386-4119-4 

- [79] KARIMI, K., N.G. DICKSON und F. HAMZE. _A Performance Comparison of CUDA and OpenCL:_ arXiv, 2010 

- [80] SELENIUMHQ. _Selenium_ [online] [Zugriff am: 27. Oktober 2022]. Verfügbar unter: https://github.com/SeleniumHQ/selenium 

- [81] LIBRARY OF CONGRESS. _STEP-file, ISO 10303-21_ [online] [Zugriff am: 27. Oktober 2022]. Verfügbar unter: 

   - https://www.loc.gov/preservation/digital/formats/fdd/fdd000448.shtml 

- [82] SCHMEH, K. _Kryptografie. Verfahren, Protokolle, Infrastrukturen._ 6., aktualisierte Auflage. Heidelberg: dpunkt.verlag, 2016. iX-Edition. ISBN 9783864903564 

114 

Literaturverzeichnis 

- [83] SZILVŚI-NAGY, M. und G. MÁTYÁSI. Analysis of STL files [online]. _Mathematical and Computer Modelling,_ 2003, **38** (7-9), S. 945-960. ISSN 08957177. Verfügbar unter: doi:10.1016/S0895-7177(03)90079-3 

- [84] QIAN-YI ZHOU, JAESIK PARK und VLADLEN KOLTUN. Open3D: A Modern Library for 3D Data Processing. _arXiv:1801.09847,_ 2018 

- [85] HUANG, J., H. SU und L. GUIBAS. _Robust Watertight Manifold Surface Generation Method for ShapeNet Models:_ arXiv, 2018 

- [86] HUANG, J., Y. ZHOU und L. GUIBAS. _ManifoldPlus: A Robust and Scalable Watertight Manifold Surface Generation Method for Triangle Soups:_ arXiv, 2020 

- [87] WU, Z., S. SONG, A. KHOSLA, F. YU, L. ZHANG, X. TANG und J. XIAO. _3D ShapeNets: A Deep Representation for Volumetric Shapes:_ arXiv, 2014 

- [88] LOOP, C. _Smooth Subdivision Surfaces Based on Triangles,_ 1987 

- [89] TAUBIN, G. Curve and surface smoothing without shrinkage. In: _Proceedings of IEEE International Conference on Computer Vision:_ IEEE Comput. Soc. Press, 20. Juni 1995, S. 852-857. ISBN 0-8186-7042-8 

- [90] SORKINE, O. und M. ALEXA. As-Rigid-as-Possible Surface Modeling. In: _Proceedings of the Fifth Eurographics Symposium on Geometry Processing._ Goslar, DEU: Eurographics Association, 2007, S. 109-116. ISBN 9783905673463 

- [91] BRUNS, N. Blender : Universelle 3D-Bearbeitungs- und Animationssoftware [online]. _Der Unfallchirurg,_ 2020, **123** (9), S. 747-750. Verfügbar unter: doi:10.1007/s00113-020-00836-0 

- [92] BLENDER. _blender_ [online] [Zugriff am: 28. Oktober 2022]. Verfügbar unter: https://github.com/blender/blender 

- [93] MAXIM TKACHENKO, MIKHAIL MALYUK, ANDREY HOLMANYUK und NIKOLAI LIUBIMOV. Label Studio: Data labeling software, 2020-2022 

- [94] LIN, T.-Y., M. MAIRE, S. BELONGIE, L. BOURDEV, R. GIRSHICK, J. HAYS, P. PERONA, D. RAMANAN, C.L. ZITNICK und P. DOLLÁR. _Microsoft COCO: Common Objects in Context:_ arXiv, 2014 

- [95] . _Sparse matrices._ New York: Academic Press, 2010. Mathematics in science and engineering. v. 99. ISBN 0126856508 

- [96] REDMON, J., S. DIVVALA, R. GIRSHICK und A. FARHADI. _You Only Look Once: Unified, Real-Time Object Detection:_ arXiv, 2015 

- [97] BOCHKOVSKIY, A., C.-Y. WANG und H.-Y.M. LIAO. _YOLOv4: Optimal Speed and Accuracy of Object Detection:_ arXiv, 2020 

115 

Literaturverzeichnis 

- [98] LI, C., L. LI, H. JIANG, K. WENG, Y. GENG, L. LI, Z. KE, Q. LI, M. CHENG, W. NIE, Y. LI, B. ZHANG, Y. LIANG, L. ZHOU, X. XU, X. CHU, X. WEI und X. WEI. _YOLOv6: A Single-Stage Object Detection Framework for Industrial Applications:_ arXiv, 2022 

- [99] DING, X., X. ZHANG, N. MA, J. HAN, G. DING und J. SUN. _RepVGG: Making VGG-style ConvNets Great Again:_ arXiv, 2021 

- [100] LIU, S., L. QI, H. QIN, J. SHI und J. JIA. _Path Aggregation Network for Instance Segmentation:_ arXiv, 2018 

- [101] BISHOP, C.M. _Pattern recognition and machine learning._ 9. (corrected at 8th printing). New York: Springer, 2009. Information science and statistics. ISBN 0387310738 

- [102] LI, L., K. JAMIESON, A. ROSTAMIZADEH, E. GONINA, M. HARDT, B. RECHT und A. TALWALKAR. A System for Massively Parallel Hyperparameter Tuning [online], 2018. Verfügbar unter: doi:10.48550/arXiv.1810.05934 

116 

Anhang A – Autopep8 Konfigurationsdatei 

## **Anhang A – Autopep8 Konfigurationsdatei** 

## _Code A.1: Konfigurationsdatei für autopep8_ 

```
[tool.autopep8]
max_line_length = 100
ignore = “E11,E121,E402”
in-place = true
indent-size = 2
aggressive = 1
recursive = true
```

117 

Anhang B – Datensatzinformationen 

## **Anhang B – Datensatzinformationen** 

_Tabelle B.1: Hersteller, von welchen versucht wurde Dateien zu aggregieren_ 

|Hersteller|||
|---|---|---|
|ABB|ABB Jokab Safety|ABL Sursum|
|ACS Kabel|Adamczewski<br>Elektronische<br>Messtechnik|Advanced Energy|
|AEG|Airtec|Alfing Montagetechnik|
|Allen Bradley|APC|Arbor|
|Argo-Hytos|Ari-Armaturen|Aski|
|ASM|ASO|Astraada|
|Automated Logic|Autosen|Axiomtek|
|B&R|Bachmann|Baco|
|Balluff|Banner Engineering|Bartec|
|Baumer Electric Ag|Baumueller|Beckhoff|
|Belimo|Bender|Benedict|
|Bernstein|Bicker Elektronik|Bieler+Lang|
|Bihl+Wiedemann|Binder|Bitner|
|Block|Bopla|Bosch Rexroth|
|Brainboxes|Brinkmann Pumpen|Bticino|
|Bulgin|Bürkert|Busch-Jaeger|
|Camille Bauer|Camtec Power Supplies|Carlo Gavazzi|
|Cg Drives & Automation|Cisco|Cliff|
|Comap|Comat Releco|Conrad|
|Cool Expert|Cosmotec|Crestron|
|Crouzet|Crydom|D+H Mechatronic|
|Dahms|Danfoss|Dehn|
|Deif|Delta Controls|Delta Electronics|
|Demag|Deos|Diggelmann|
|Dina Elektronik|Di-Soric|Dixell|
|Doepke|Dold|Drago|
|E.T.A. S.P.A.|EAE|EAO|



118 

Anhang B – Datensatzinformationen 

|Hersteller|||
|---|---|---|
||||
||||
|EAP Electric|Eaton|Eberle|
|EES|Eldon|Elektra Tailfingen|
|Elesta|Elitra|Elmeko|
|Elobau|Elster|Eltako|
|Emerson|Enda|Endress&Hauser|
|Enertex Bayern|Engel Elektroantriebe|EPA|
|Erhardt+Leimer|Escha|Esser|
|ETA|ETA|Euchner|
|Eurotherm|Evon Home|Ewon|
|Faber Kabel|Festo|Fibox|
|Fiessler Elektronik|Finder|Finsecur|
|Fipa|Frako|Fränkische|
|Frogblue|Fuehlersysteme Enet<br>International|Futek|
|Gantner Instruments|GE|GE Fanuc|
|GE Intelligent Platforms|Georg Schlegel|Getriebebau Nord|
|Gira|Global Control 5|Gmc-I Messtechnik|
|Graesslin|Grundfos|Hager|
|Harting|Häwa|HBC-Radiomatic Gmbh|
|HBM|Heco|Heidenhain|
|Helmholz|Helukabel|Hematec|
|Hengstler|Hensel|Hewlett-Packard|
|Hima|Hitachi|Hms Industrial Networks|
|Hoerbiger|Honeywell|Honigmann|
|Horner Apg|Hydac|IAI|
|Icotek|Icp Das|Idec|
|IDS|IFM|Igus|
|Ilme|Insevis|Inter Control|
|Iskra|ITT Corporation|J. Schneider|
|J+J Deutschland|Janitza|Jean Müller|
|Jetter|Johnson Controls|Jumo|
|Jung|Kaleja|Kbr|



119 

Anhang B – Datensatzinformationen 

|Hersteller|||
|---|---|---|
||||
||||
|Keb|Keyence|Kfm|
|Kieback&Peter|Kistler|Knick|
|Koch|Kollmorgen|Koncar|
|Konzept Energietechnik|Kora Industrie-Elektronik|Kraus & Naimer|
|Kries Energietechnik|Kübler|Kukla Waagenfabrik|
|Kunbus|Lapp|Lcn - Issendorff|
|Legrand|Lemo|Lenze|
|Leutron|Leuze|Lewa|
|LG|Lingg - Janke|Lohmeier|
|Lovato Electric|Loxone|Loytec|
|LS Industrial Systems|LTI Motion|Lumberg Automation|
|Lunatone Industrielle<br>Elektronik Gmbh|Lütze|Maintronic|
|Marposs|Martens|Mastervolt|
|Matsushita|MB Connect Line|MBS Ag|
|MDT Technologies|Mean Well|Mehler|
|Mennekes|Mersen|Merten|
|Metronix|Metz Connect|Mitsubishi|
|Motor Power Company|Motorola|Moxa|
|MP Elettronica|MPM|Murrelektronik|
|MVA Vertriebs Gmbh|Mygekko|Nanotec|
|National Instruments|Neutrik|Nidec|
|Niedax|Norgren|Novatek-Electro|
|Novotechnik|OBO Bettermann|OEZ|
|OHP Automation Systems|Omron|Oppermann Regelgeräte|
|Oriental Motor|Panasonic|Parker|
|Pepperl+Fuchs|Peter Electronic|Pfannenberg|
|Phoenix|Pilz|Pinter|
|Pizzato Elettrica|Powerio|Pr Electronics|
|Priva|Prominent|Puls|
|Pulsotronic|Raritan|Raspberry Pi|
|RDM|Reco|Rehau|



120 

Anhang B – Datensatzinformationen 

|Hersteller|||
|---|---|---|
||||
||||
|Relmatic|Relpol|Reo Ag|
|Rittal|Ritto|RK-Tec|
|Romutec|Rössel-Messtechnik|Rübsamen & Herr|
|Rutenbeck|S + S Regeltechnik|Satel|
|Sauter|SBA|SBC|
|Schaffner|Schalk|Schmersal|
|Schneider Electric|Schrack|Scs Static Control<br>Systems|
|SHE|Sensus|Serai|
|Servotronix|SEW|Sick|
|Siemens|Sigmatek|Sirio Elettronica|
|Siteco|Sitron Sensor|SMC|
|Socomec|Solvimus|Spelsberg|
|SSP|Stahl|Stasto Automation|
|Steffel KKS|Stego|Stoeber|
|Störk-Tronic|Striebel & John|Sulzer|
|Sun Hydraulics|Tapeswitch|Technische Alternative<br>RT|
|Technokabel|Tecnotion|Tele Haase|
|Telegärtner|Tempa Pano|Theben|
|Thermokon|Tkd|Trafomodern|
|TR-Electronic|Trend Controls|Trendnet|
|Trumeter|Turck|Unitronics|
|Universal Robots|UWT|Vacon|
|Valbia|Vega|Vem Motors|
|Victron Energy|Vipa|Wachendorff<br>Prozesstechnik|
|Wagner Magnete|Wago|Walther-Werke|
|Warema|Watlow|WEG|
|Weidmueller|Weintek|Weiss|
|Wenglor|Werma|Wieland|
|Wiesemann & Theis|Wiska|Wöhner|
|Woodward|Wunderli Electronics Ag|Yaskawa|



121 

Anhang B – Datensatzinformationen 

|Hersteller|||
|---|---|---|
||||
||||
|Ypc|Zennio Avance Y<br>Tecnología|Ziehl|
|Zimmer Automation|Zimmer Group|Zpas|



_Tabelle B.2: Herstellerbezogene Informationen zu den Häufigkeiten der Dateien im gesamten Datensatz_ 

|Hersteller|STEP|Hash|OBJ/OFF|Labels|
|---|---|---|---|---|
|ABB|52|31|14|8|
|Allen Bradley|9|6|2|1|
|Balluff|4|4|1|0|
|Beckhoff|101|29|1|1|
|Bicker Elektronik|1|1|1|0|
|Bopla|14|7|0|0|
|Bosch Rexroth|14|5|0|0|
|B&R|64|22|6|0|
|Brainboxes|9|6|0|0|
|Bulgin|1|1|0|0|
|Camtec Power Supplies|15|5|4|0|
|Carlo Gavazzi|65|5|2|2|
|Cliff|1|1|1|0|
|ComAp|21|6|1|0|
|Crouzet|3|2|0|0|
|Danfoss|20|6|0|0|
|Dehn|545|183|51|28|
|Dina Elektronik|2|2|0|0|
|EAE|4|4|0|0|
|Eaton|358|106|19|0|
|Eldon|2|2|0|0|
|Elitra|1|1|0|0|
|ETA|16|1|1|0|



122 

Anhang B – Datensatzinformationen 

|Hersteller|STEP|Hash|OBJ/OFF|Labels|
|---|---|---|---|---|
|ETI|1984|152|0|0|
|Eurotherm|1|1|0|0|
|Finder|1|1|0|0|
|Grundfos|1|1|0|0|
|Häwa|360|301|0|0|
|Hensel|14|9|0|0|
|IAI|1|1|0|0|
|IDEC|6|6|6|0|
|IFM|7|6|3|0|
|Igus|1|1|0|0|
|Janitza|1|11|0|0|
|Jean Müller|1|1|1|0|
|Kaleja|36|15|2|0|
|KEB|1|1|1|0|
|Keyence|7|7|2|0|
|Kistler|2|2|0|0|
|Kunbus|20|8|0|0|
|Lenze|288|84|43|0|
|Lovato Electric|4|4|3|0|
|LS Industrial Systems|27|6|1|0|
|Mastervolt|125|73|5|0|
|Mean Well|1|1|1|0|
|Mitsubishi|110|42|5|0|
|Murrelektronik|34|23|7|0|
|Novatek-Electro|27|21|11|0|
|OBO Bettermann|1|1|0|0|
|Omron|64|57|4|0|
|Phoenix|10264|10202|3588|151|



123 

Anhang B – Datensatzinformationen 

|Hersteller|STEP|Hash|OBJ/OFF|Labels|
|---|---|---|---|---|
|Pilz|79|18|3|0|
|PR Electronics|1|1|0|0|
|Puls|12|10|1|0|
|Schmersal|1|1|0|0|
|Schneider Electric|1255|951|41|0|
|SEW|3|3|2|0|
|Siemens|28242|7799|689|16|
|Sigmatek|4|3|0|0|
|SMC|12|5|0|0|
|Socomec|6|1|0|0|
|Stasto Automation|4|3|0|0|
|Stoeber|133|3|0|0|
|Sun Hydraulics|1|1|1|0|
|Tecnotion|3|3|0|0|
|Tempa Pano|426|426|0|0|
|Trafomodern|10|9|8|0|
|Turck|3|3|3|0|
|Unitronics|22|3|0|0|
|Wago|969|329|163|23|
|Weidmueller|62|39|15|4|
|Weintek|13|8|0|0|
|Wenglor|1|1|0|0|
|Wieland|1|38|6|0|
|Wiesemann & Theis|2|2|0|0|



124 

Anhang B – Datensatzinformationen 

_Tabelle B.3: Kategorien des Datensatzes, welche gefiltert und nicht weiter berücksichtigt wurden_ 

|Kategorie 1|Kategorie 2|Kategorie 3|
|---|---|---|
|Cabinet Engineering|||
|Not Categorized|||
|Fluid Engineering|||
|Electrical Installation|||
|Eletrical Engineering|Cables and Conductors||
||Measuring+reporting<br>devices||
||KNX (EIB)||
||Power Source||
||Loads||
||Switch+Protection|Accessories|
||Contactor, Relays|Accessories|
||Plugs|Accessories|
||Terminals|Accessories|



125 

Anhang B – Datensatzinformationen 

**==> picture [454 x 630] intentionally omitted <==**

**----- Start of picture text -----**<br>
Wiesemann & Theis<br>Wieland<br>Wenglor<br>Weintek<br>Weidmueller<br>Wago<br>Unitronics<br>Turck<br>Trafomodern<br>Tempa Pano<br>Tecnotion<br>Sun Hydraulics<br>Stoeber<br>Stasto Automation<br>Socomec<br>SMC<br>Sigmatek<br>Siemens<br>SEW<br>Schneider Electric<br>Schmersal<br>Puls<br>PR Electronics<br>Pilz<br>Phoenix<br>Omron<br>OBO Bettermann<br>Novatek-Electro<br>Murrelektronik<br>Mitsubishi<br>Mean Well<br>Mastervolt<br>LS Industrial Systems<br>Lovato Electric<br>Lenze<br>Kunbus<br>Kistler<br>Keyence<br>KEB<br>Kaleja<br>Jean Müller<br>Janitza<br>Igus<br>IFM<br>IDEC<br>IAI<br>Hensel<br>Häwa<br>Grundfos<br>Finder<br>Eurotherm<br>ETI<br>ETA<br>Elitra<br>Eldon<br>Eaton<br>EAE<br>Dina Elektronik<br>Dehn<br>Danfoss<br>Crouzet<br>ComAp<br>Cliff<br>Carlo Gavazzi<br>Camtec Power Supplies<br>Bulgin<br>Brainboxes<br>Bosch Rexroth<br>Bopla<br>Bicker Elektronik<br>Beckhoff<br>Balluff<br>B&R<br>Allen Bradley<br>ABB<br>0 10 20 30 40 50 60 70 80 % 100<br>Anteil der Unikate →<br>Konvertierungsrate (STL zu OBJ) Unikate<br>Hersteller<br>**----- End of picture text -----**<br>


_Abbildung B.1: Prozentuale Rate der Unikate je Hersteller und Konvertierungsrate von STL- zu OBJ-Dateien_ 

126 

Anhang C – Verwendete Softwareversionen 

## **Anhang C – Verwendete Softwareversionen** 

_Tabelle C.1: Auflistung der wichtigsten verwendeten Python-Pakete inklusive Softwareversion, falls diese nicht in den entsprechenden Kapiteln ausdrücklich definiert wurden._ 

|Software|Version|
|---|---|
|autopep8|1.7.0|
|bayesian-optimization|1.2.0|
|ipykernel|6.9.1|
|ipython|8.4.0|
|numpy|1.22.03|
|opencv-python|4.6.0.66|
|pillow|9.0.1|
|pip|22.1.2|
|plotly|5.9.0|
|pythonocc-core|7.6.2|
|pytorch|1.11.0|
|ray|1.13.0|
|scikit-learn|1.1.1|
|scipy|1.8.1|
|torchmetrics|0.10.0|
|torchsummary|1.5.1|
|torchvision|0.12.0|
|tqdm|4.64.0|



127 

Anhang C – Verwendete Softwareversionen 

_Tabelle C2: Verwendete Software aus Git-Repositories_ 

|Software|Version|Branch|
|---|---|---|
|bensch98/cad-scraper|git@github.com:bensch98/cad-<br>scraper.git|main|
|bensch98/ccc-dataset|git@github.com:bensch98/ccc-<br>dataset.git|main|
|bensch98/control-cabinet|git@github.com:bensch98/control-<br>cabinet.git|main|
|blender/blender|git@github.com:blender/blender.git|v3.3.0|
|hjwdzh/Manifold|git@github.com:hjwdzh/Manifold.git|master|
|hjwdzh/ManifoldPlus|git@github.com:hjwdzh/ManifoldPlus.g<br>it|master|
|LSnyd/MedMeshCNN|git@github.com:LSnyd/MedMeshCNN.<br>git|master|
|meituan/YOLOv6|git@github.com:meituan/YOLOv6.git|main|
|nmwsharp/diffusion-net|git@github.com:nmwsharp/diffusion-<br>net.git|master|
|ranahanocka/MeshCNN|git@github.com:ranahanocka/MeshCN<br>N.git|master|



128 

Anhang D – Digitaler Anhang 

## **Anhang D – Digitaler Anhang** 

## **Datenträger mit:** 

- Digitale Version der Arbeit 

- Citavi-Datei 

- Abbildungen 

- Git-Repositories 

- CAD-Modelle 

- Datenauswertung 

- HDD mit Datensatz + Metainformationen 

## BENEDIKT SCHEFFLER 

## **BILDUNG** 

## **Master Wirtschaftsingenieurwesen (Maschinenbau)** 

2020 – heute (4. Semester) Friedrich-Alexander-Universität Erlangen-Nürnberg Aktueller Schnitt: 1,5 

## **Bachelor Wirtschaftsingenieurwesen (Maschinenbau)** 

2017/18 – 2020 

Friedrich-Alexander-Universität Erlangen-Nürnberg Finale Note: 1,9 

## **Gymnasium Höchstadt a. d. Aisch** 

## **PERSÖNLICHE DATEN** 

GEBURTSTAG: 26.03.1998 

GEBURTSORT: Bamberg, Deutschland NATIONALITÄT: Deutsch 

2008 – 2016 Abitur: 2,1 

## **BERUFLICHE ERFAHRUNG** 

## **FAPS** 

Studentische Hilfskraft – Machine Learning Operations – Elektromaschinenbau Mai 2022 – heute 

## **CARIAD SE** 

## **KONTAKT** 

ADRESSE: Graslitzer Str. 8a 91315 Höchstadt a. d. Aisch 

TELEFON: +49 171 8123924 

E-MAIL: scheffler.benedikt@gmail.com 

## LinkedIn 

Praktikum – Cloud Software Engineering – Chassis Functions November 2021 – April 2022 

## **Siemens Healthineers** 

Werkstudent/Praktikant – Software Development – Process & Tool Support / Business Management August 2019 – September 2021 

## **Elektrobit Automotive** 

Werkstudent – Project Validation Engineering Juli 2017 – Juli 2019 

## **QUALIFIKATIONEN** 

**Sprachen:** Deutsch, Englisch **IT-Kenntnisse:** Linux (Ubuntu, Debian), Kubernetes, Docker, MS Office 

## **Programmierkenntnisse:** 

Python, C, C++, Go, bash, SQL, NoSQL (MongoDB) 

