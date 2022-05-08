
library(shiny)
library(shinythemes)
install.packages("tidyquant")
library(tidyquant)
install.packages("dygraphs")
library(dygraphs)
library("readxl")
library(datasets)

### UI ###

ui <- fluidPage( theme = shinytheme("cosmo"),
                 
                 ## Logo Header
                 fluidRow(
                   column(8, offset = 2, align = "center",
                          img(src="logo_korian.png", width=400)
                   )
                 ),
                 fluidRow(
                   column(8, offset = 2, align = "center",
                   h1("Les Fossoyeurs case study and comparison with Korian reports")
                   )
                 ),
                 
                 ## testo introduttivo
                 fluidRow(
                   column(10, offset = 1,
                          #titlePanel("ORPEA case study and comparison with Korian reports"),
                          p("Public opinion is crucial to the performance of healthcare management companies. When it comes to health services, people and their trust are at the heart of the business.
                 From an economic point of view, there have been recent examples of the enormous impact of public opinion on the market."),
                          p("On January 22, 2022, the investigative book Les Fossoyeurs by journalist Victor Castanet was published. 
                          He described a system with frightening aspects in the nursing homes managed by the french company ORPEA, a worldwide leader in the management of healthcare facilities. 
                          In addition to the financial and bureaucratic aspects of the business, Castanet highlighted in the book the lack of care in the more human aspects of the sector. 
                          Lack of hygiene, poor medical care, rationed meals and unjustifiably high prices."),
                          p("Shortly after the publication, the general manager of the group Yves Le Masne was replaced. 
                            In the following two weeks, the shares on the stock market fell by 62%, which equates to a loss of euro 56 billions."),
                          p("The fall of the global leader has dragged down other companies operating in the same sector, including Korian. 
                            For this reason, it is key analyzing the complaints to understand which services to improve, or study the trend of complaints after some decision-making processes."),
                          p("Also, the complaints analysis is crucial to defend the business reputation during media attacks as happened in France this year. 
                            That is, to highlight the work done for the services improvement with respect to the complaints received."),
                          br(),
                          br(),
                          column(8, offset = 2, align = "center",
                            img(src="trend_febr_kor_orp.png", width=600)
                          ),
                          ),
                 ),
                 br(),
                 hr(),
                 
                 ## Testo intro alle info
                 fluidRow(
                   column(10, offset = 1,
                          h2("What do Korian customers complain about? Are they the same ones denounced in the book Les Fossoyeurs?"),
                          p("A comparison between the aspects denounced in the book Les Fossoyeurs and an insight into what Korian’s customers appreciate or complain about the most."),
                          br(),
                          br(),
                   )
                 ),
                 
                 ## Parte con Info tabs ====================================================================================================================
                 
                 fluidRow(
                   column( width= 10,offset= 1,
                           
                           navlistPanel(widths=c(2,10),
                             "Reclami",
                             tabPanel("Unigram", img(src="Top_20_reclami_unigram.png", width= "100%")),
                             tabPanel("Trigram", img(src="Top_20_reclami_trigram.png", width= "100%")),
                             tabPanel("Fivegram", img(src="Top_20_reclami_5gram.png", width= "100%"))
                           ),
                           hr(),
                   )
                 ),
                 
                 fluidRow(
                   column( width= 10,offset= 1,
                           
                           navlistPanel(widths=c(2,10),
                             "Apprezzamenti",
                             tabPanel("Unigram", img(src="Top_20_apprezzamenti_unigram.png", width= "100%")),
                             tabPanel("Trigram", img(src="Top_20_apprezzamenti_trigram.png", width= "100%")),
                             tabPanel("Fivegram", img(src="Top_20_apprezzamenti_5gram.png", width= "100%"))
                           ),
                           hr(),
                   )
                 ),
                 
                
                 
                 fluidRow(
                   column( width= 10,offset= 1,
                   h2("Topics in Les Fossoyeurs"),
                   p("Select the Chapter"),
                   ),
                   br(),
                   column( width= 10,offset= 1, align="center",
                           tabsetPanel(
                             #h4("Select the Chapter"),
                             #br(),
                             #br(),
                             tabPanel(title = "Intro", img(src="LF_top20_Intro.png", width= "100%") ),
                             tabPanel(title = "1", img(src="LF_top20_Chap_1.png", width= "100%") ),
                             tabPanel(title = "2", img(src="LF_top20_Chap_2.png", width= "100%") ),
                             tabPanel(title = "3", img(src="LF_top20_Chap_3.png", width= "100%") ),
                             tabPanel(title = "4", img(src="LF_top20_Chap_4.png", width= "100%") ),
                             tabPanel(title = "5", img(src="LF_top20_Chap_5.png", width= "100%") ),
                             tabPanel(title = "6", img(src="LF_top20_Chap_6.png", width= "100%") ),
                             tabPanel(title = "7", img(src="LF_top20_Chap_7.png", width= "100%") ),
                             tabPanel(title = "8", img(src="LF_top20_Chap_8.png", width= "100%") ),
                             tabPanel(title = "9", img(src="LF_top20_Chap_9.png", width= "100%") ),
                             tabPanel(title = "10", img(src="LF_top20_Chap_10.png", width= "100%") ),
                             tabPanel(title = "11", img(src="LF_top20_Chap_11.png", width= "100%") ),
                             tabPanel(title = "12", img(src="LF_top20_Chap_12.png", width= "100%") ),
                             tabPanel(title = "13", img(src="LF_top20_Chap_13.png", width= "100%") ),
                             tabPanel(title = "14", img(src="LF_top20_Chap_14.png", width= "100%") ),
                             tabPanel(title = "15", img(src="LF_top20_Chap_15.png", width= "100%") ),
                             tabPanel(title = "16", img(src="LF_top20_Chap_16.png", width= "100%") ),
                             tabPanel(title = "17", img(src="LF_top20_Chap_17.png", width= "100%") ),
                             tabPanel(title = "18", img(src="LF_top20_Chap_18.png", width= "100%") ),
                             tabPanel(title = "19", img(src="LF_top20_Chap_19.png", width= "100%") ),
                             tabPanel(title = "20", img(src="LF_top20_Chap_20.png", width= "100%") ),
                             tabPanel(title = "21", img(src="LF_top20_Chap_21.png", width= "100%") ),
                             tabPanel(title = "22", img(src="LF_top20_Chap_22.png", width= "100%") ),
                             tabPanel(title = "23", img(src="LF_top20_Chap_23.png", width= "100%") ),
                             tabPanel(title = "24", img(src="LF_top20_Chap_24.png", width= "100%") ),
                             tabPanel(title = "25", img(src="LF_top20_Chap_25.png", width= "100%") ),
                             tabPanel(title = "26", img(src="LF_top20_Chap_26.png", width= "100%") ),
                             tabPanel(title = "27", img(src="LF_top20_Chap_27.png", width= "100%") ),
                             tabPanel(title = "28", img(src="LF_top20_Chap_28.png", width= "100%") ),
                             tabPanel(title = "29", img(src="LF_top20_Chap_29.png", width= "100%") ),
                             tabPanel(title = "30", img(src="LF_top20_Chap_30.png", width= "100%") ),
                             tabPanel(title = "31", img(src="LF_top20_Chap_31.png", width= "100%") ),
                             tabPanel(title = "32", img(src="LF_top20_Chap_32.png", width= "100%") ),
                             tabPanel(title = "33", img(src="LF_top20_Chap_33.png", width= "100%") ),
                             tabPanel(title = "34", img(src="LF_top20_Chap_34.png", width= "100%") ),
                             tabPanel(title = "35", img(src="LF_top20_Chap_35.png", width= "100%") ),
                             tabPanel(title = "36", img(src="LF_top20_Chap_36.png", width= "100%") ),
                             tabPanel(title = "37", img(src="LF_top20_Chap_37.png", width= "100%") ),
                             tabPanel(title = "38", img(src="LF_top20_Chap_38.png", width= "100%") ),
                           )
                           
                   )
                 ),
                 
                 # =====================================================================================================================================================
                 
                 fluidRow(
                   column(10, offset = 1,
                          h2("Which is the trend in time of the complaints' severity per service?"),
                          p("A view of the severity trend of the feedback received in recent years."),
                          br(),
                          br(),
                   )
                 ),
                 
                 
                 fluidRow(
                   column( width= 10,offset= 1,
                           
                           navlistPanel( fluid = TRUE, widths=c(2,10),
                                         "Capitoli",
                                         tabPanel("Nursing",
                                                  img(src="Polarity_trend_NursingCare.png", width= "100%")
                                         ),
                                         tabPanel("Medical Area",
                                                  img(src="Polarity_trend_MedicalArea.png", width= "100%")
                                         ),
                                         tabPanel("Management",
                                                  img(src="Polarity_trend_Management.png", width= "100%")
                                         ),
                                         tabPanel("Hospitality",
                                                  img(src="Polarity_trend_Hospitality.png", width= "100%")
                                         ),
                                         tabPanel("Catering",
                                                  img(src="Polarity_trend_Catering.png", width= "100%")
                                         ),
                                         tabPanel("Other Services",
                                                  img(src="Polarity_trend_OtherServices.png", width= "100%")
                                         ),
                           ),
                           br(),
                           hr(),
                           br(),
                   )
                 ),
                 
                 
                 
                 
                 # ======================================================================================================================================
                 
                 ## Footer
                 fluidRow(
                   column( width=2, offset= 5,
                           actionLink(inputId= "githublink", label="Go to GitHub", icon=icon("github", "fa-2x"), onclick ="window.open('https://github.com/c-majer/NLP_Brand_Reputation_LesFossoyeurs_Korian', '_blank')"),
                           br(), br(), br()
                   )
                 )
)

### SERVER ### #############################################################################################################################################

server <- function(input, output) {
  
  
  
}

### RUN THE APP ########################################################################################################################################

shinyApp(ui = ui, server = server)
