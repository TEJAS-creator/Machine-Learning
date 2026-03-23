=========================
LAB 1 – UNIX COMMANDS
=========================

# 1. Create directory and subdirectory
mkdir RegNo
cd RegNo
mkdir Lab1

# 2. Execute basic commands and save output
pwd > output.txt
ls >> output.txt
date >> output.txt
whoami >> output.txt
echo Hello >> output.txt

# 3. File operations
touch file1.txt
cat > file1.txt
Hello
Ctrl+D

cp file1.txt copy.txt
mv file1.txt file2.txt
rm file2.txt

# 4. Pattern matching
ls *.txt
ls *[0-9]*
ls ????*
ls [!aeiouAEIOU]*


=========================
LAB 2 – ADVANCED UNIX
=========================

# grep
grep apple file.txt
grep -i apple file.txt
grep -v apple file.txt
grep '^[A-Z]' file.txt
grep '\.$' file.txt

# sort
sort file.txt
sort -r file.txt
sort -n file.txt

# wc
wc file.txt
wc -l file.txt
wc -w file.txt

# cut
cut -c1-3 file.txt
cut -d':' -f1 file.txt

# tr
tr '[a-z]' '[A-Z]' < file.txt

# sed
sed 's/apple/orange/' file.txt
sed 's/apple/orange/g' file.txt


=========================
LAB 3 – SHELL SCRIPTING
=========================

# 1. File type check
#!/bin/bash
echo "Enter file name:"
read f

if [ -d "$f" ]
then
    echo "Directory"
elif [ -f "$f" ]
then
    echo "File"
else
    echo "Not exists"
fi


# 2. List files with pattern
#!/bin/bash
echo "Enter folder:"
read folder
echo "Enter pattern:"
read pattern

grep -l "$pattern" $folder/*


# 3. Rename .txt to .text
#!/bin/bash
find . -type f -name "*.txt" | while read f
do
    mv "$f" "${f%.txt}.text"
done


# 4. Gross salary
#!/bin/bash
echo "Enter basic:"
read basic
echo "Enter TA:"
read ta

gs=$(echo "$basic + $ta + 0.10*$basic" | bc)
echo "Gross Salary=$gs"


# 5. Copy files by extension
#!/bin/bash
echo "Enter extension:"
read ext
echo "Enter folder:"
read folder

mkdir -p $folder

for f in *.$ext
do
    cp "$f" "$folder"
done


# 6. Replace ex:
#!/bin/bash
for f in *
do
    if [ -f "$f" ]
    then
        sed -i 's/^ex:/Example:/g; s/\. ex:/\. Example:/g' "$f"
    fi
done


# 7. Delete even lines
#!/bin/bash
echo "Enter file:"
read f
sed '2~2d' "$f"


=========================
LAB 4 – PROCESS & THREADS
=========================

# 1. fork + wait
#include<stdio.h>
#include<unistd.h>
#include<sys/wait.h>

int main()
{
    pid_t pid=fork();

    if(pid==0)
    {
        printf("Child PID=%d PPID=%d\n",getpid(),getppid());
    }
    else
    {
        wait(NULL);
        printf("Parent PID=%d Child=%d\n",getpid(),pid);
    }
}


# 2. exec
#include<stdio.h>
#include<unistd.h>

int main()
{
    if(fork()==0)
    {
        execl("./a.out","a.out",NULL);
    }
}


# 3. Zombie
#include<stdio.h>
#include<unistd.h>
#include<stdlib.h>

int main()
{
    if(fork()==0)
        exit(0);
    else
        sleep(20);
}


# 4. Multithread sorting
#include<stdio.h>
#include<pthread.h>

int a[100],n;

void *sort()
{
    for(int i=0;i<n-1;i++)
        for(int j=0;j<n-i-1;j++)
            if(a[j]>a[j+1])
            {
                int t=a[j];
                a[j]=a[j+1];
                a[j+1]=t;
            }
}

int main()
{
    pthread_t t;
    printf("Enter n: ");
    scanf("%d",&n);

    for(int i=0;i<n;i++)
        scanf("%d",&a[i]);

    pthread_create(&t,NULL,sort,NULL);
    pthread_join(t,NULL);

    for(int i=0;i<n;i++)
        printf("%d ",a[i]);
}


# 5. Fibonacci thread
#include<stdio.h>
#include<pthread.h>

int f[100],n;

void *fib()
{
    f[0]=0;
    f[1]=1;

    for(int i=2;i<n;i++)
        f[i]=f[i-1]+f[i-2];
}

int main()
{
    pthread_t t;
    printf("Enter n: ");
    scanf("%d",&n);

    pthread_create(&t,NULL,fib,NULL);
    pthread_join(t,NULL);

    for(int i=0;i<n;i++)
        printf("%d ",f[i]);
}


=========================
LAB 5 – CPU SCHEDULING
=========================

# 1. FCFS
#include<stdio.h>

int main()
{
    int n;
    printf("Enter number of processes: ");
    scanf("%d",&n);

    int at[n],bt[n],ct[n],wt[n],tat[n];

    for(int i=0;i<n;i++)
    {
        printf("AT BT for P%d: ",i+1);
        scanf("%d%d",&at[i],&bt[i]);
    }

    int time=0;

    for(int i=0;i<n;i++)
    {
        if(time<at[i]) time=at[i];

        time+=bt[i];
        ct[i]=time;

        tat[i]=ct[i]-at[i];
        wt[i]=tat[i]-bt[i];
    }

    printf("P\tCT\tTAT\tWT\n");
    for(int i=0;i<n;i++)
        printf("P%d\t%d\t%d\t%d\n",i+1,ct[i],tat[i],wt[i]);
}


# 2. SRTF
#include<stdio.h>

int main()
{
    int n;
    scanf("%d",&n);

    int at[n],bt[n],rt[n],ct[n];

    for(int i=0;i<n;i++)
    {
        scanf("%d%d",&at[i],&bt[i]);
        rt[i]=bt[i];
    }

    int complete=0,t=0;

    while(complete<n)
    {
        int min=999,pos=-1;

        for(int i=0;i<n;i++)
            if(at[i]<=t && rt[i]>0 && rt[i]<min)
            {
                min=rt[i];
                pos=i;
            }

        if(pos==-1){ t++; continue; }

        rt[pos]--;
        t++;

        if(rt[pos]==0)
        {
            ct[pos]=t;
            complete++;
        }
    }

    for(int i=0;i<n;i++)
        printf("P%d CT=%d\n",i+1,ct[i]);
}


# 3. Round Robin
#include<stdio.h>

int main()
{
    int n,q;
    scanf("%d",&n);

    int bt[n],rt[n],ct[n];

    for(int i=0;i<n;i++)
    {
        scanf("%d",&bt[i]);
        rt[i]=bt[i];
    }

    scanf("%d",&q);

    int time=0,done;

    do
    {
        done=1;

        for(int i=0;i<n;i++)
        {
            if(rt[i]>0)
            {
                done=0;

                if(rt[i]>q)
                {
                    time+=q;
                    rt[i]-=q;
                }
                else
                {
                    time+=rt[i];
                    ct[i]=time;
                    rt[i]=0;
                }
            }
        }
    }while(!done);

    for(int i=0;i<n;i++)
        printf("P%d CT=%d\n",i+1,ct[i]);
}


=========================
LAB 6 – IPC
=========================

# 1. Message Queue
#include<stdio.h>
#include<sys/ipc.h>
#include<sys/msg.h>

struct msg{ long type; int num; };

int main()
{
    int id=msgget(1234,0666|IPC_CREAT);
    struct msg m;

    m.type=1;

    printf("Enter number:");
    scanf("%d",&m.num);

    msgsnd(id,&m,sizeof(m),0);
}


# 2. FIFO
#include<stdio.h>
#include<fcntl.h>
#include<unistd.h>
#include<sys/stat.h>

int main()
{
    int fd,data[4];

    mkfifo("fifo",0666);
    fd=open("fifo",O_WRONLY);

    for(int i=0;i<4;i++)
        scanf("%d",&data[i]);

    write(fd,data,sizeof(data));
}


# 3. Shared Memory
#include<stdio.h>
#include<sys/ipc.h>
#include<sys/shm.h>

int main()
{
    int id=shmget(1234,100,0666|IPC_CREAT);
    char *p=shmat(id,NULL,0);

    printf("Enter char:");
    scanf(" %c",p);

    *p=*p+1;

    printf("Next char=%c",*p);
}


# 4. Shared Memory Producer Consumer
#include<stdio.h>
#include<sys/ipc.h>
#include<sys/shm.h>

int main()
{
    int id=shmget(1234,1024,0666|IPC_CREAT);
    char *p=shmat(id,NULL,0);

    printf("Enter text:");
    fgets(p,1024,stdin);

    printf("Read:%s",p);
}
