public class main{
    public static void main(String[] args){
        for(int i = 0; i < 5; i++){
            if(i == 0 || i == 4){
                System.out.println("*".repeat(5));
            }
            else{
                System.out.println(" ".repeat(4-i) + "*");
            }
        }
    }
}
